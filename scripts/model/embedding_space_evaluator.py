import time

import numpy as np
import torch
import torch.nn.functional as F
import umap
from scipy import linalg

from model.embedding_net import EmbeddingNet
from model.motion_ae import MotionAE

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore warnings


class EmbeddingSpaceEvaluator:
    def __init__(self, args, embed_net_path, lang_model, device):
        self.n_pre_poses = args.n_pre_poses

        # init embed net
        ckpt = torch.load(embed_net_path, map_location=device)
        n_frames = args.n_poses
        word_embeddings = lang_model.word_embedding_weights
        mode = 'pose'
        self.pose_dim = ckpt['pose_dim']

        if args.pose_dim == 27:
            self.net = EmbeddingNet(args, self.pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                    word_embeddings, mode).to(device)
            self.net.load_state_dict(ckpt['gen_dict'])
        elif args.pose_dim == 126:
            self.latent_dim = ckpt['latent_dim']
            self.net = MotionAE(self.pose_dim, self.latent_dim).to(device)
            self.net.load_state_dict(ckpt['motion_ae'])

        self.net.train(False)

        # storage
        self.context_feat_list = []
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []

        self.cos_err_diff = []

    def reset(self):
        self.context_feat_list = []
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []

        self.cos_err_diff = []

    def get_no_of_samples(self):
        return len(self.real_feat_list)

    def push_samples(self, context_text, context_spec, generated_poses, real_poses):
        # convert poses to latent features
        if self.pose_dim == 27:
            pre_poses = real_poses[:, 0:self.n_pre_poses]
            context_feat, _, _, real_feat, _, _, real_recon = self.net(context_text, context_spec, pre_poses, real_poses,
                                                                    'pose', variational_encoding=False)
            _, _, _, generated_feat, _, _, generated_recon = self.net(None, None, pre_poses, generated_poses,
                                                                    'pose', variational_encoding=False)
            if context_feat:
                self.context_feat_list.append(context_feat.data.cpu().numpy())
        elif self.pose_dim == 126:
            real_recon, real_feat = self.net(real_poses)
            generated_recon, generated_feat = self.net(generated_poses)

        self.real_feat_list.append(real_feat.data.cpu().numpy())
        self.generated_feat_list.append(generated_feat.data.cpu().numpy())

        # # reconstruction error
        # recon_err_real = F.l1_loss(real_poses, real_recon).item()
        # recon_err_fake = F.l1_loss(generated_poses, generated_recon).item()
        # self.recon_err_diff.append(recon_err_fake - recon_err_real)

        reconstruction_loss_real = F.l1_loss(real_recon, real_poses, reduction='none')
        reconstruction_loss_real = torch.mean(reconstruction_loss_real, dim=(1, 2))

        if True:  # use pose diff
            target_diff = real_poses[:, 1:] - real_poses[:, :-1]
            recon_diff = real_recon[:, 1:] - real_recon[:, :-1]
            reconstruction_loss_real += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

        reconstruction_loss_real = torch.sum(reconstruction_loss_real)
        cos_loss_real = torch.sum(1 - torch.cosine_similarity(real_recon.view(real_recon.shape[0], real_recon.shape[1], -1, 3), real_poses.view(real_poses.shape[0], real_poses.shape[1], -1, 3), dim = -1))

        reconstruction_loss_fake = F.l1_loss(generated_recon, generated_poses, reduction='none')
        reconstruction_loss_fake = torch.mean(reconstruction_loss_fake, dim=(1, 2))

        if True:  # use pose diff
            target_diff = generated_poses[:, 1:] - generated_poses[:, :-1]
            recon_diff = generated_recon[:, 1:] - generated_recon[:, :-1]
            reconstruction_loss_fake += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

        reconstruction_loss_fake = torch.sum(reconstruction_loss_fake)
        cos_loss_fake = torch.sum(1 - torch.cosine_similarity(generated_recon.view(generated_recon.shape[0], generated_recon.shape[1], -1, 3), generated_poses.view(generated_poses.shape[0], generated_poses.shape[1], -1, 3), dim = -1))

        self.recon_err_diff.append(reconstruction_loss_fake - reconstruction_loss_real)
        self.cos_err_diff.append(cos_loss_fake - cos_loss_real)

    def get_features_for_viz(self):
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)

        transformed_feats = umap.UMAP().fit_transform(np.vstack((generated_feats, real_feats)))
        n = int(transformed_feats.shape[0] / 2)
        generated_feats = transformed_feats[0:n, :]
        real_feats = transformed_feats[n:, :]

        return real_feats, generated_feats

    def get_diversity_scores(self):
        feat1 = np.vstack(self.generated_feat_list[:500])
        random_idx = torch.randperm(len(self.generated_feat_list))[:500]
        shuffle_list = [self.generated_feat_list[x] for x in random_idx]
        feat2 = np.vstack(shuffle_list)
        # dists = []
        # for i in range(feat1.shape[0]):
        #     d = np.sum(np.absolute(feat1[i] - feat2[i]))  # MAE
        #     dists.append(d)
        feat_dist = np.mean(np.sum(np.absolute(feat1 - feat2), axis=-1))
        return feat_dist

    def get_scores(self):
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)

        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)
            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10
            return frechet_dist

        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)