import torch
import torch.nn as nn
import numpy as np
class EstimatorCV():    # estimate covariance
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def save(self, root='./output'):
        _cov_pd = np.array(self.CoVariance.cpu(), dtype=np.float)
        _ave_pd = np.array(self.Ave.cpu(), dtype=np.float)
        cov = pd.DataFrame(_cov_pd)
        ave = pd.DataFrame(_ave_pd)
        print('cov:', cov.shape)
        print('ave:', ave.shape)
        print(cov[0])
        cov.to_csv('{0}/cov.csv'.format(root), header=True, index=False)
        ave.to_csv('{0}/ave_svhn.csv'.format(root), header=True, index=False)

    def update_CV(self, features, labels):  # feature = (N, A)
        N = features.size(0)    # batchsize
        C = self.class_num  # 10 for cifar10
        A = features.size(1)    # length of feature vector

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA
        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)
        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)
        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A))
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(sum_weight_AV + self.Amount.view(C, 1).expand(C, A))
        weight_AV[weight_AV != weight_AV] = 0
        # right term
        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )
        # left term + right term
        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(weight_CV)).detach() + additional_CV.detach()
        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()
        self.Amount += onehot.sum(0)


class ISDALoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(ISDALoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()

    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0] # 最后的FC层参数（bs, feature_num, class_num)
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels.view(N, 1, 1).expand(N, C, A))
        CV_temp = cv_matrix[labels]

        # sigma2 = ratio * \
        #          torch.bmm(torch.bmm(NxW_ij - NxW_kj,
        #                              CV_temp).view(N * C, 1, A),
        #                    (NxW_ij - NxW_kj).view(N * C, A, 1)).view(N, C)

        sigma2 = ratio * torch.bmm(torch.bmm(NxW_ij - NxW_kj, CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)
        aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, model, fc, x, target, ratio):

        features = model(x) # feature before fc layers
        y = fc(features)    # raw-prediction
        self.estimator.update_CV(features.detach(), target)
        isda_aug_y = self.isda_aug(fc, features, y, target, self.estimator.CoVariance.detach(), ratio)
        loss = self.cross_entropy(isda_aug_y, target)

        return loss, y

