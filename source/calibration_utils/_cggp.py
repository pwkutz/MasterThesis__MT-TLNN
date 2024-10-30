# This code was kindly sent to me by Jonathan Wenger. It is not my own!!!

"""Conjugate Gradient-based Gaussian processes."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import itergp
import probnum
from itergp import methods
from probnum import backend, linops, problems, randprocs, randvars
from probnum.linalg import solvers


class ConjugateGradientGaussianProcess(randprocs.GaussianProcess):
    """CG-based approximation of a Gaussian process conditioned on data.

    Parameters
    ----------
    X
        Input data.
    y
        Output data.
    b
        Observation noise.
    maxiter
        Maximum number of iterations for CG.
    """

    def __init__(
        self,
        prior: randprocs.GaussianProcess,
        X: backend.Array,
        Y: backend.Array,
        b: randvars.Normal,
        maxiter: int,
        atol=1e-6,
        rtol=1e-6,
    ):
        if prior.output_shape != ():
            raise ValueError("Currently, only scalar conditioning is supported.")

        X, Y, pred_mean_X, gram_XX = self._preprocess_observations(
            prior=prior, X=X, Y=Y, b=b
        )

        self._prior = prior
        self._X = X
        self._Y = Y

        self._pred_mean_X = pred_mean_X
        self._gram_XX = gram_XX

        self._maxiter = maxiter
        self._atol = atol
        self._rtol = rtol

        self._k_X: Callable[[backend.Array], backend.Array] = lambda x: self._prior.cov(
            backend.expand_dims(x, axis=-self._prior.input_ndim - 1),
            self._X,
        )

        # self._representer_weights = backend.asarray(
        #     gpytorch.utils.linear_cg(
        #         torch.as_tensor(self._gram_XX, dtype=torch.double),
        #         torch.as_tensor(self._Y, dtype=torch.double),
        #         tolerance=self._atol,
        #         max_iter=self._maxiter,
        #         max_tridiag_iter=self._maxiter,
        #     ),
        #     dtype=backend.float64,
        # )

        # Compute representer weights
        P = linops.Zero(shape=(X.shape[0], X.shape[0]))
        Pinv = linops.Zero(shape=(X.shape[0], X.shape[0]))
        problem = problems.LinearSystem(gram_XX, Y - pred_mean_X)
        linsys_prior = solvers.beliefs.LinearSystemBelief(
            x=randvars.Normal(mean=Pinv @ problem.b, cov=linops.aslinop(gram_XX).inv()),
            Ainv=Pinv,
            A=P,
        )
        qoi_belief, _ = methods.CG(
            maxiter=self._maxiter, atol=self._atol, rtol=self._rtol
        ).solve(problem=problem, prior=linsys_prior)
        self._representer_weights = qoi_belief.x.mean

        super().__init__(
            mean=ConjugateGradientGaussianProcess.Mean(
                prior=self._prior,
                X=self._X,
                representer_weights=self._representer_weights,
            ),
            cov=ConjugateGradientGaussianProcess.Kernel(
                prior_kernel=self._prior.cov,
                X=self._X,
                k_X=self._k_X,
                gram_XX=self._gram_XX,
                atol=self._atol,
                rtol=self._rtol,
                maxiter=self._maxiter,
            ),
        )

    class Mean(probnum.Function):
        """Mean function of a Gaussian process conditioned on data.

        Parameters
        ----------
        prior
            Gaussian process prior.
        X
            Input data.
        representer_weights
            Representer weights :math:`\hat{K}^{-1}y`.
        """

        def __init__(
            self,
            prior: randprocs.GaussianProcess,
            X: Tuple[backend.Array],
            representer_weights: randvars.RandomVariable,
        ) -> None:
            self._prior = prior
            self._X = X
            self._representer_weights = representer_weights

            super().__init__(
                input_shape=self._prior.input_shape,
                output_shape=self._prior.output_shape,
            )

        def _evaluate(self, x: backend.Array) -> backend.Array:
            m_x = self._prior.mean(x)
            k_x_X = self._prior.cov.linop(x, self._X)

            return m_x + k_x_X @ self._representer_weights

    class Kernel(randprocs.kernels.Kernel):
        """Kernel of a SVGP approximation."""

        def __init__(
            self,
            prior_kernel: randprocs.kernels.Kernel,
            X: backend.Array,
            k_X: Callable[[backend.Array], backend.Array],
            gram_XX: backend.Array,
            atol: float,
            rtol: float,
            maxiter: int,
        ):
            self._prior_kernel = prior_kernel
            self._X = X
            self._k_X = k_X
            self._gram_XX = gram_XX
            self._atol = atol
            self._rtol = rtol
            self._maxiter = maxiter

            super().__init__(
                input_shape=prior_kernel.input_shape,
                output_shape=prior_kernel.output_shape,
            )

        def _evaluate(
            self, x0: backend.Array, x1: Optional[backend.Array] = None
        ) -> backend.Array:
            k_xx = self._prior_kernel(x0, x1)
            k_x0_X = self._k_X(x0)
            k_x1_X = self._k_X(x1) if x1 is not None else k_x0_X

            # linearsolve = backend.asarray(
            #     gpytorch.utils.linear_cg(
            #         torch.as_tensor(self._gram_XX, dtype=torch.double),
            #         torch.as_tensor(k_x1_X, dtype=torch.double).transpose(-1, -2),
            #         tolerance=self._atol,
            #         max_iter=self._maxiter,
            #         max_tridiag_iter=self._maxiter,
            #     ).transpose(-1, -2),
            #     dtype=backend.float64,
            # )[..., :, None]

            # return k_xx - (k_x0_X[..., None, :] @ linearsolve)[..., 0, 0]

            P = linops.Zero(shape=self._gram_XX.shape)
            Pinv = linops.Zero(shape=self._gram_XX.shape)
            problem = problems.LinearSystem(
                self._gram_XX, self._prior_kernel(self._X, self._X[0, ...])
            )
            linsys_prior = solvers.beliefs.LinearSystemBelief(
                x=randvars.Normal(
                    mean=Pinv @ problem.b, cov=linops.aslinop(self._gram_XX).inv()
                ),
                Ainv=Pinv,
                A=P,
            )
            qoi_belief, _ = methods.CG(
                maxiter=self._maxiter, atol=self._atol, rtol=self._rtol
            ).solve(problem=problem, prior=linsys_prior)
            gram_XX_inv = qoi_belief.Ainv

            return (
                k_xx
                - (k_x0_X[..., None, :] @ (gram_XX_inv @ k_x1_X[..., :, None]))[
                    ..., 0, 0
                ]
            )

    @classmethod
    def _preprocess_observations(
        cls,
        prior: randprocs.GaussianProcess,
        X: backend.Array,
        Y: backend.Array,
        b: Optional[Union[randvars.Normal, randvars.Constant]],
    ) -> Tuple[backend.Array, backend.Array, backend.Array, backend.Array,]:
        # Reshape to (N, input_dim) and (N,)
        X = backend.asarray(X)
        Y = backend.asarray(Y)

        assert prior.output_shape == ()
        assert (
            X.ndim >= 1 and X.shape[X.ndim - prior._input_ndim :] == prior.input_shape
        )
        assert Y.shape == X.shape[: X.ndim - prior._input_ndim] + prior.output_shape

        X = X.reshape((-1,) + prior.input_shape, order="C")
        Y = Y.reshape((-1,), order="C")

        # Apply measurement operator to prior
        f = prior
        k = prior.cov

        # Compute predictive mean and kernel Gram matrix
        pred_mean_X = f.mean(X)
        gram_XX = itergp.linops.KernelMatrix(kernel=k, x0=X)

        if b is not None:
            assert isinstance(b, (randvars.Constant, randvars.Normal))
            assert b.shape == Y.shape

            pred_mean_X += b.mean.reshape((-1,), order="C")
            # This assumes that the covariance matrix is raveled in C-order
            gram_XX += linops.aslinop(b.cov)

        return X, Y, pred_mean_X, gram_XX
