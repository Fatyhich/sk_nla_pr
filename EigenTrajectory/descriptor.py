import time
import torch
import torch.nn as nn
import numpy as np
from .normalizer import TrajNorm
from scipy.linalg import svd as classic_svd
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds as iterative_svd


class ETDescriptor(nn.Module):
    r"""EigenTrajectory descriptor model

    Args:
        hyper_params (DotDict): The hyper-parameters
        norm_ori (bool): Whether to normalize the trajectory with the origin
        norm_rot (bool): Whether to normalize the trajectory with the rotation
        norm_sca (bool): Whether to normalize the trajectory with the scale"""

    def __init__(self, hyper_params, norm_ori=True, norm_rot=True, norm_sca=True):
        super().__init__()

        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.traj_normalizer = TrajNorm(ori=norm_ori, rot=norm_rot, sca=norm_sca)

        self.U_obs_trunc = nn.Parameter(torch.zeros((self.t_obs * self.dim, self.k)))
        self.U_pred_trunc = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.k)))

    def normalize_trajectory(self, obs_traj, pred_traj=None):
        r"""Trajectory normalization

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (Optional, for training only)

        Returns:
            obs_traj_norm (torch.Tensor): The normalized observed trajectory
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
        """

        self.traj_normalizer.calculate_params(obs_traj)
        obs_traj_norm = self.traj_normalizer.normalize(obs_traj)
        pred_traj_norm = self.traj_normalizer.normalize(pred_traj) if pred_traj is not None else None
        return obs_traj_norm, pred_traj_norm

    def denormalize_trajectory(self, traj_norm):
        r"""Trajectory denormalization

        Args:
            traj_norm (torch.Tensor): The trajectory to be denormalized

        Returns:
            traj (torch.Tensor): The denormalized trajectory
        """

        traj = self.traj_normalizer.denormalize(traj_norm)
        return traj

    def to_ET_space(self, traj, evec):
        r"""Transform Euclidean trajectories to EigenTrajectory coefficients

        Args:
            traj (torch.Tensor): The trajectory to be transformed
            evec (torch.Tensor): The ET descriptors (eigenvectors)

        Returns:
            C (torch.Tensor): The ET descriptor coefficients"""

        # Euclidean -> ET
        tdim = evec.size(0)
        M = traj.reshape(-1, tdim).T
        C = evec.T.detach() @ M
        return C

    def to_Euclidean_space(self, C, evec):
        r"""Transform EigenTrajectory coefficients to Euclidean trajectories

        Args:
            C (torch.Tensor): The ET descriptor coefficients
            evec (torch.Tensor): The ET descriptors (eigenvectors)

        Returns:
            traj (torch.Tensor): The Euclidean trajectory"""

        # ET -> Euclidean
        t = evec.size(0) // self.dim
        M = evec.detach() @ C
        traj = M.T.reshape(-1, t, self.dim)
        return traj


    def truncated_SVD(self, traj, k=None, svd_method='torch', full_matrices=False):
        r"""Truncated Singular Value Decomposition

        Args:
            traj (torch.Tensor): The trajectory to be decomposed
            k (int): The number of singular values and vectors to be computed
            full_matrices (bool): Whether to compute full-sized matrices

        Returns:
            U_trunc (torch.Tensor): The truncated left singular vectors
            S_trunc (torch.Tensor): The truncated singular values
            Vt_trunc (torch.Tensor): The truncated right singular vectors
        """

        assert traj.size(2) == self.dim  # NTC
        k = self.k if k is None else k

        # Singular Value Decomposition
        M = traj.reshape(-1, traj.size(1) * self.dim).T

        print(f'choosed solver: {svd_method}')
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        if svd_method == 'torch':
            U, S, Vt = torch.linalg.svd(M, full_matrices=full_matrices)
        elif svd_method == 'torch_cpu':
            traj_cpu = traj.cpu()
            M = traj_cpu.reshape(-1, traj_cpu.size(1) * self.dim).T
            with torch.no_grad():
                torch.set_default_device('cpu')
                U, S, Vt = torch.linalg.svd(M)

        elif svd_method == 'scipy':
            M_np = M.cpu().detach().numpy()
            U, S, Vt = classic_svd(M_np, full_matrices=True)
            U = torch.tensor(U).to(traj.device)
            S = torch.tensor(S).to(traj.device)
            Vt = torch.tensor(Vt).to(traj.device)
            print('Back to torch success')
        elif svd_method == 'sparse':
            M_np = M.cpu().detach().numpy()
            U_trunc, S_trunc, Vt_trunc = iterative_svd(M_np, k=k)
            U_trunc = torch.tensor(U_trunc).to(traj.device)
            S_trunc = torch.tensor(np.flip(S_trunc).copy()).to(traj.device)
            Vt_trunc = torch.tensor(np.flip(Vt_trunc).copy()).to(traj.device)
            
            end = time.perf_counter() - start
            print('Back to torch success')
            print(f'Iterative SVD time (s) spent : {end}')
            print(f'Original Matrix shape : {M.shape}')
            print(f'Truncated SVD limit: {k}')

            return U_trunc, S_trunc, Vt_trunc.T, end
        elif svd_method == 'random':
            M_np = M.cpu().detach().numpy()
            U_trunc, S_trunc, Vt_trunc = iterative_svd(M_np, k=k)
            U_trunc = torch.tensor(U_trunc).to(traj.device)
            S_trunc = torch.tensor(np.flip(S_trunc).copy()).to(traj.device)
            Vt_trunc = torch.tensor(np.flip(Vt_trunc).copy()).to(traj.device)

            end = time.perf_counter() - start
            print('Back to torch success')
            print(f'Iterative SVD time (s) spent : {end}')
            print(f'Original Matrix shape : {M.shape}')
            print(f'Truncated SVD limit: {k}')

            return U_trunc, S_trunc, Vt_trunc.T, end

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter() - start
        print(f'Full SVD time (s) spent : {end}')
        print(f'Original Matrix shape : {M.shape}')
        print(f'SVD (aka USVt) shape : {U.shape}, \t, {S.shape} \t {Vt.shape}')

        # Truncated SVD
        U_trunc, S_trunc, Vt_trunc = U[:, :k], S[:k], Vt[:k, :]
        print(f'Truncation parameter k= {k}')

        return U_trunc, S_trunc, Vt_trunc.T, end
    
    def time_test(self, obs_traj, pred_traj, cycle_number, svd_method='torch'):
        r"""SVD Time Test

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory
            cycle_nummber (int): How much cycle of SVD you have

        Returns:
            Avarage: Time for

        """
        # Normalize trajectory
        obs_traj_norm, pred_traj_norm = self.normalize_trajectory(obs_traj, pred_traj)
        timeStorage = []

        for i in range(0, cycle_number):
            # Singular Value Decomposition with truncation
            _, _, _, t = self.truncated_SVD(obs_traj_norm, svd_method=svd_method)
            timeStorage.append(t)
            _, _, _, t = self.truncated_SVD(pred_traj_norm, svd_method=svd_method)
            timeStorage.append(t)

        average_time = sum(timeStorage) / len(timeStorage)

        print(f'Average Time for {cycle_number} iterations with {svd_method} method is {average_time}')

        # Reuse values for anchor generation
        return average_time


    def parameter_initialization(self, obs_traj, pred_traj, svd_method='torch'):
        r"""Initialize the ET descriptor parameters (for training only)

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory

        Returns:
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
            U_pred_trunc (torch.Tensor): The truncated eigenvectors of the predicted trajectory

        Note:
            This function should be called once before training the model."""

        # Normalize trajectory
        obs_traj_norm, pred_traj_norm = self.normalize_trajectory(obs_traj, pred_traj)

        # Singular Value Decomposition with truncation
        U_obs_trunc, _, _, _ = self.truncated_SVD(obs_traj_norm, svd_method=svd_method)
        U_pred_trunc, _, _, _ = self.truncated_SVD(pred_traj_norm, svd_method=svd_method)

        # Register eigenvectors as model parameters
        self.U_obs_trunc = nn.Parameter(U_obs_trunc.to(self.U_obs_trunc.device))
        self.U_pred_trunc = nn.Parameter(U_pred_trunc.to(self.U_pred_trunc.device))

        # Reuse values for anchor generation
        return pred_traj_norm, U_pred_trunc

    def projection(self, obs_traj, pred_traj=None):
        r"""Trajectory projection to the ET space

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (optional, for training only)

        Returns:
            C_obs (torch.Tensor): The observed trajectory in the ET space
            C_pred (torch.Tensor): The predicted trajectory in the ET space (optional, for training only)
        """

        # Trajectory Projection
        obs_traj_norm, pred_traj_norm = self.normalize_trajectory(obs_traj, pred_traj)
        C_obs = self.to_ET_space(obs_traj_norm, evec=self.U_obs_trunc).detach()
        C_pred = self.to_ET_space(pred_traj_norm, evec=self.U_pred_trunc).detach() if pred_traj is not None else None
        return C_obs, C_pred

    def reconstruction(self, C_pred):
        r"""Trajectory reconstruction from the ET space

        Args:
            C_pred (torch.Tensor): The predicted trajectory in the ET space

        Returns:
            pred_traj (torch.Tensor): The predicted trajectory in the Euclidean space
        """

        # Trajectory Reconstruction
        pred_traj_norm = [self.to_Euclidean_space(C_pred[:, :, s], evec=self.U_pred_trunc) for s in range(self.s)]
        pred_traj = [self.denormalize_trajectory(pred_traj_norm[s]) for s in range(self.s)]
        pred_traj = torch.stack(pred_traj, dim=0)  # SNTC
        return pred_traj

    def forward(self, C_pred):
        r"""Alias for reconstruction"""

        return self.reconstruction(C_pred)
