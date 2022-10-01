from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from emp.env import Environment
from emp.robot import Robot

np.set_printoptions(precision=5, suppress=True, linewidth=300)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Optimizer(nn.Module):
    def __init__(self, opt):
        super(Optimizer, self).__init__()
        self.environment = Environment(opt)

        MRS = []
        for i in range(opt.size_MRS):
            MRS.append(Robot(self.environment.graph, i + 1, opt))
        self.robots = MRS

        self.curPolicy = None
        self.stationaryDistributionBuffer, self.transitionMatrixBuffer = [], []
        if opt.variational:
            distribution = list(map(float, input(f'\nPlease input your expected distribution for ' + opt.env + ':\n').strip().split()))[:len(self.environment.nodes)]
            self.user_defined_distribution = list(np.array(distribution) / np.sum(np.array(distribution)))
            assert len(self.user_defined_distribution) == len(self.environment.nodes), \
                opt.env + f'has {len(self.environment.nodes)}. However, the length of your distribution is {len(self.user_defined_distribution)}!'

    def reset(self):
        self.curPolicy = None
        self.stationaryDistributionBuffer.clear()
        self.transitionMatrixBuffer.clear()

    def forward(self):
        # Get transition matrix
        transitionMatrix = torch.stack([self.curPolicy(self.curPolicy.generateStateVector(i)) for i in range(self.curPolicy.numS)]).to(self.curPolicy.device)
        assert not torch.isnan(transitionMatrix).any(), f'{transitionMatrix}'
        assert transitionMatrix.ge(0.0).all() and transitionMatrix.le(1.0).all(), f'{transitionMatrix}'

        # Calculate stationary distribution
        PIE = torch.linalg.inv(transitionMatrix
                               - torch.eye(self.curPolicy.numS).to(self.curPolicy.device)
                               + torch.ones(self.curPolicy.numS, self.curPolicy.numS).to(self.curPolicy.device))
        stationaryDistribution = PIE.sum(0)
        assert not torch.isnan(stationaryDistribution).any(), f'{stationaryDistribution}'
        assert stationaryDistribution.ge(0.0).all() and stationaryDistribution.le(1.0).all(), f'{stationaryDistribution}'

        return stationaryDistribution, transitionMatrix

    def getOthersList(self, k):
        d_others = [ts.detach() for ts in self.stationaryDistributionBuffer]
        P_others = [mat.detach() for mat in self.transitionMatrixBuffer]
        d_others.pop(k)
        P_others.pop(k)
        d_others = torch.stack(d_others)
        P_others = torch.stack(P_others)

        return d_others, P_others

    def printHistory(self, history, opt):
        plt.suptitle(opt.env + f', {len(self.robots)} robot(s), $\\alpha_r,\,\\alpha_s,\,\\alpha,\,\epsilon=${opt.alpha_r}, {opt.alpha_s}, {opt.lr}, {opt.epsilon}')

        plt.subplot(2, 2, 1)
        plt.plot(np.array(history).T[0][0], label=f'MRS')
        plt.legend(loc='lower right')
        plt.ylabel('$J\,(\widetilde{J})+\\alpha_r \cdot J_r + \\alpha_s \cdot J_s$')

        plt.subplot(2, 2, 2)
        if opt.variational:
            KLDmin = 0.0 * np.ones(shape=(np.array(history).shape[0],), dtype=float)
            plt.plot(KLDmin, 'g--', label='Min KLD')
            plt.plot(-np.array(history).T[1][0], label=f'MRS')
            plt.ylabel('$\widetilde{J}$')
        else:
            plt.ylabel('$J$')
            plt.plot(np.array(history).T[1][0], label=f'MRS')
        plt.legend(loc='lower right')


        plt.subplot(2, 2, 3)
        Hmax = np.log(len(self.environment.nodes)) * np.ones(shape=(np.array(history).shape[0],), dtype=float)
        plt.plot(Hmax, 'g--', label='Max Ent')
        plt.plot(np.array(history).T[2][0], label=f'MRS')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('$J_r$')

        plt.subplot(2, 2, 4)
        plt.plot(np.array(history).T[3][0], label=f'MRS')
        plt.legend(loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('$J_s$')

        plt.tight_layout()
        plt.savefig(opt.env + f'_MRS({opt.size_MRS})_Epoch({opt.epoch})_AlphaR({opt.alpha_r})_AlphaS({opt.alpha_s}).png', dpi=300)

    def trainPolicy(self, opt):
        epoch = opt.epoch
        history = []
        best_obj = -100000000

        # Initialization
        for k in range(len(self.robots)):
            self.curPolicy = self.robots[k].policy
            print(f'The optimizer is creating buffer for robot {self.curPolicy.idx}!')
            stationaryDistribution, transitionMatrix = self.forward()
            self.stationaryDistributionBuffer.append(stationaryDistribution)
            self.transitionMatrixBuffer.append(transitionMatrix)

        # Updating curPolicy parameters in receding horizon
        trainProgress = tqdm(total=epoch)
        for i in range(epoch):
            history_epoch = []
            trainProgress.update(1)
            this_epoch_obj = 0

            for k in range(len(self.robots)):
                # Choose a curPolicy to update
                self.curPolicy = self.robots[k].policy
                d, P = self.stationaryDistributionBuffer[k], self.transitionMatrixBuffer[k]
                d_absent_log = torch.log(torch.ones(1, self.curPolicy.numS).to(self.curPolicy.device) - d)
                Ent = -torch.dot(d, torch.log(d))
                Ent_Rate = torch.stack([torch.dot(P[s] * -1.0, torch.log(P[s])) for s in range(self.curPolicy.numS)])
                Ent_Rate = torch.dot(d, Ent_Rate)

                J_r, J_s = Ent, Ent_Rate
                if len(self.robots) > 1:
                    d_others, P_others = self.getOthersList(k)
                    Ent_others = -torch.sum(torch.stack([torch.dot(d_others[idx], torch.log(d_others)[idx]) for idx in range(len(d_others))]))
                    Ent_Rate_others = torch.zeros(1, requires_grad=True).to(self.curPolicy.device)[0]
                    for idx in range(len(P_others)):
                        Ent_Rate_others += torch.dot(d_others[idx], torch.stack([torch.dot(P_others[idx][s] * -1.0, torch.log(P_others[idx][s])) for s in range(self.curPolicy.numS)]))
                    d_absent_log += torch.log(torch.ones_like(d_others).to(self.curPolicy.device) - d_others).sum(dim=0)
                    J_r += Ent_others
                    J_s += Ent_Rate_others
                J_r /= float(len(self.robots))
                J_s /= float(len(self.robots))

                d_overall = (torch.ones(1, self.curPolicy.numS).to(self.curPolicy.device) - torch.exp(d_absent_log)).squeeze(0)

                if opt.variational:
                    d_overall /= torch.sum(d_overall.detach())
                    expectedD = torch.Tensor(self.user_defined_distribution).to(self.curPolicy.device)
                    J = -F.kl_div(torch.log(expectedD), d_overall, reduction='sum')

                else:
                    J = torch.dot(d_overall * -1.0, torch.log(d_overall + self.curPolicy.epsilon * torch.ones_like(d_overall).to(self.curPolicy.device)))

                obj = -(J + opt.alpha_r * J_r + opt.alpha_s * J_s)

                # Update parameters
                self.curPolicy.optimizer.zero_grad()
                obj.backward()
                self.curPolicy.optimizer.step()

                # Check gradients
                for name, params in self.curPolicy.named_parameters():
                    assert torch.isfinite(params.grad).all(), f'{name}: {params.grad}'
                '''
                    print('-->name:', name)
                    print('-->params:', params)
                    print('-->grad_requires:', params.requires_grad)
                    print('-->grad_value:', params.grad)
                '''

                history_epoch.append([float(-obj.detach()), float(J.detach()), float(Ent.detach()), float(Ent_Rate.detach())])
                if opt.verbose:
                    if (i + 1) % (epoch / 10) == 0:
                        print('=' * 100)
                        print(f'At epoch {i + 1}, the values of robot {k + 1} are:\n'
                              f'Obj: {round(float(-obj.cpu().detach().numpy()), 3)}='
                              f'{round(float(J.cpu().detach().numpy()), 3)}'
                              f'+{opt.alpha_r}*{round(float(Ent.cpu().detach().numpy()), 3)}'
                              f'+{opt.alpha_s}*{round(float(Ent_Rate.cpu().detach().numpy()), 3)},\n'
                              f'J: {round(float(J.cpu().detach().numpy()), 3)},\n'
                              f'J_r: {round(float(Ent.cpu().detach().numpy()), 3)} ({0.0 if opt.variational else round(float(np.log(self.curPolicy.numS)), 3)} max.),\n'
                              f'J_s: {round(float(Ent_Rate.cpu().detach().numpy()), 3)}.')
                        print(f'At epoch {i + 1}, the stationary distribution of robot {k + 1} is:\n'
                              f'{d.cpu().detach().numpy()}, stddev={round(float(np.std(d.cpu().detach().numpy())), 3)}')
                        print(f'At epoch {i + 1}, the transition matrix of robot {k + 1} is:')
                        for s in range(self.curPolicy.numS):
                            print(f'State {s + 1}: {P[s].cpu().detach().numpy()}')
                        print('=' * 100)
                # Update buffer
                self.stationaryDistributionBuffer[k], self.transitionMatrixBuffer[k] = self.forward()
                this_epoch_obj = -obj

            history.append(history_epoch)
            if opt.output and this_epoch_obj >= best_obj:
                for robot in self.robots:
                    torch.save(robot.policy.state_dict(), opt.env + f'_MRS({opt.size_MRS})_robot{robot.policy.idx}.pkl')
                best_obj = this_epoch_obj

        trainProgress.close()
        if opt.output:
            for robot in self.robots:
                np.savetxt(opt.env + f'_MRS({opt.size_MRS})_robot{robot.policy.idx}_history.txt', np.array(history)[:, 0, :].T)
        self.reset()
        self.printHistory(history, opt)

    def testPolicy(self, opt):
        step = opt.step
        for R in self.robots:
            R.initPosition(opt)

        testProgress = tqdm(total=step)
        for i in range(step):
            testProgress.update(1)
            visitRecord_t = np.zeros(shape=(len(self.environment.nodes),), dtype=int)
            for R in self.robots:
                s_t = R.getCurrentNode()
                D_a_t = R.policy(R.policy.generateStateVector(s_t - 1))
                a_t = R.selectAction(D_a_t, s_t)
                R.executeAction(a_t)
                if visitRecord_t[a_t - 1] == 0: visitRecord_t[a_t - 1] = 1
                R.updateVisitHistory(R.getCurrentNode())
            self.environment.updateVisitFrequency(visitRecord_t)
        testProgress.close()

        print(('*' * 50) + 'EMP' + ('*' * 50))
        for R in self.robots:
            freq_R = R.getVisitFrequency()
            print(f'Robot {R.policy.idx}\'s visit frequency within {step} steps:\n'
                  f'{freq_R}')
            R.reset()
        freq_E = self.environment.getVisitFrequency(opt)
        print(f'The environment\'s visit frequency of {len(self.robots)} robot(s) within {step} steps:\n'
              f'{freq_E * opt.step}')
        if opt.variational:
            freq = list(freq_E * opt.step)
            freq /= np.sum(freq)
            KLD = F.kl_div(torch.log(torch.Tensor(self.user_defined_distribution)), torch.Tensor(freq), reduction='sum').detach().numpy()
            print(f'~J: {round(float(KLD), 3)} (0.0 min.)')
        else:
            print(f'J: {round(-1.0 * np.dot(freq_E, np.log(freq_E)), 3)}')
        print(('*' * 50) + 'EMP' + ('*' * 50))
        self.environment.reset()


