"""Fourier Delay-SINDy: continuous delay optimization with PySINDy.

Dependencies: numpy, scipy, pysindy (tested interfaces documented in PySINDy 2.x).

This is an offline drift/derivative regression model, NOT a causal DDE integrator.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pysindy as ps
from scipy.signal import savgol_filter


class FourierDelaySINDy:
    """PySINDy-style sparse identification of a single shared continuous delay.

    Parameters
    ----------
    degree : int
        Degree of PySINDy's PolynomialLibrary.
    poly_on : {'both', 'current'}
        'both': polynomial features in [x(t), x(t-tau)], including interactions;
        'current': polynomial in x(t), plus a pure delayed term.
    delay_function, delay_derivative : callable or None
        For scalar x, optional h(z), dh(z)/dz with z=x(t-tau).
        In 'current' mode defaults to h(z)=z.
        In 'both' mode custom h(z) is appended to the polynomial library.
    delay_bounds : (float, float) or None
        Optimization box if specified. None still enforces 0 < tau < max_delay,
        as larger delays cannot be identified reliably from cropped finite data.
    max_delay : float or None
        Largest trustworthy delay for which training boundary samples are dropped.
        If None, uses upper delay bound; if both absent, 10% of record duration.
    cutoff_period : float or None
        Keep FFT modes with periods >= cutoff_period. None keeps all modes
        except the Nyquist mode, which is removed for noninteger rFFT shifts.
    threshold : float
        Relative STLSQ threshold, scaled by max absolute coefficient from
        normalized least squares. Selection runs on normalized columns.
    n_starts : int
        Number of Adam starts, including tau0 if given.
    n_alternations : int
        Number of [STLSQ support update, fixed-support delay refinement] cycles.
    selection : {'bic', 'mse'}
        Candidate selection across multistart solutions.
    """

    def __init__(
        self,
        degree: int = 3,
        poly_on: str = "both",
        feature_library=None,
        optimizer=None,
        delay_function: Optional[Callable] = None,
        delay_derivative: Optional[Callable] = None,
        delay_feature_name: str = "h(x_delay)",
        delay_bounds=(0.1, 40.0),
        max_delay: Optional[float] = None,
        tau0: Optional[float] = None,
        n_starts: int = 8,
        cutoff_period: Optional[float] = 2.0,
        sg_window: int = 51,
        sg_order: int = 3,
        threshold: float = 0.15,
        n_alternations: int = 2,
        adam_lr: float = 0.08,
        adam_iters: int = 300,
        selection: str = "bic",
        random_state: Optional[int] = 42,
    ):
        if poly_on not in ("both", "current"):
            raise ValueError("poly_on must be 'both' or 'current'")
        if selection not in ("bic", "mse"):
            raise ValueError("selection must be 'bic' or 'mse'")
        if (delay_function is None) != (delay_derivative is None):
            raise ValueError("Specify both delay_function and delay_derivative")
        if n_starts < 1 or n_alternations < 0:
            raise ValueError("n_starts >= 1 and n_alternations >= 0 required")
        self.degree = degree
        self.poly_on = poly_on
        self.feature_library = feature_library
        self.optimizer = optimizer
        self.delay_function = delay_function
        self.delay_derivative = delay_derivative
        self.delay_feature_name = delay_feature_name
        self.delay_bounds = delay_bounds
        self.max_delay = max_delay
        self.tau0 = tau0
        self.n_starts = n_starts
        self.cutoff_period = cutoff_period
        self.sg_window = sg_window
        self.sg_order = sg_order
        self.threshold = threshold
        self.n_alternations = n_alternations
        self.adam_lr = adam_lr
        self.adam_iters = adam_iters
        self.selection = selection
        self.random_state = random_state

    @staticmethod
    def _to_2d(x):
        a = np.asarray(x, dtype=float)
        if a.ndim == 1:
            return a[:, None]
        if a.ndim != 2:
            raise ValueError("x must be (n_samples,) or (n_samples, n_states)")
        return a

    @staticmethod
    def _get_dt(t, n):
        if np.ndim(t) == 0:
            dt = float(t)
        else:
            tt = np.asarray(t, dtype=float)
            if tt.shape != (n,):
                raise ValueError("t must be scalar dt or a time array of length n")
            dt = float(np.mean(np.diff(tt)))
            if not np.allclose(np.diff(tt), dt, rtol=1e-5, atol=1e-10):
                raise ValueError("FFT shift requires a uniformly sampled time series")
        if dt <= 0:
            raise ValueError("dt must be positive")
        return dt

    def fit(self, x, t=1.0, x_dot=None, feature_names=None):
        """Fit model. Return self (like pysindy.SINDy.fit).

        x_dot may be supplied externally; otherwise Savitzky-Golay estimates it.
        For process noise, this is only an approximate drift regression.
        """
        raw = self._to_2d(x)
        n, d = raw.shape
        self.dt_ = self._get_dt(t, n)
        self.n_states_ = d
        self._names = feature_names or [f"x{i}" for i in range(d)]
        if len(self._names) != d:
            raise ValueError("feature_names length must match state dimension")
        if self.delay_function is not None and d != 1:
            raise ValueError("custom delay_function currently requires scalar x")

        win = int(self.sg_window)
        if win % 2 == 0 or win <= self.sg_order or win > n:
            raise ValueError("sg_window must be odd, > sg_order, <= n_samples")
        xs = savgol_filter(raw, win, self.sg_order, axis=0)
        if x_dot is None:
            yd = savgol_filter(raw, win, self.sg_order,
                               deriv=1, delta=self.dt_, axis=0)
        else:
            yd = self._to_2d(x_dot)
            if yd.shape != raw.shape:
                raise ValueError("x_dot must have same shape as x")

        total_time = (n - 1) * self.dt_
        if self.delay_bounds is not None:
            lo, hi = map(float, self.delay_bounds)
            if lo <= 0 or hi <= lo:
                raise ValueError("delay_bounds must satisfy 0 < lo < hi")
        else:
            lo, hi = None, None
        max_delay = (float(self.max_delay) if self.max_delay is not None else
                     (hi if hi is not None else 0.10 * total_time))
        if max_delay <= 0 or (hi is not None and max_delay < hi):
            raise ValueError("max_delay must be positive and >= delay_bounds[1]")
        self.max_delay_ = max_delay
        # Fixed crop throughout optimization. Avoids inventing pre-record history.
        self.buffer_ = int(np.ceil(max_delay / self.dt_)) + win // 2 + 2
        if 2 * self.buffer_ + 25 >= n:
            raise ValueError("Record too short for max_delay and SG window")
        self.valid_indices_ = np.arange(self.buffer_, n - self.buffer_)
        self.x_smoothed_ = xs
        self.x_dot_ = yd
        self.y_ = yd[self.valid_indices_]

        ext = np.concatenate([xs, xs[-2:0:-1]], axis=0)
        self._extended_length = ext.shape[0]
        self._spectrum = np.fft.rfft(ext, axis=0)
        self._omega = 2 * np.pi * np.fft.rfftfreq(ext.shape[0], self.dt_)
        self._mask = np.ones_like(self._omega)
        if self.cutoff_period is not None:
            if self.cutoff_period <= 0:
                raise ValueError("cutoff_period must be positive")
            self._mask = (self._omega <= 2*np.pi/self.cutoff_period).astype(float)
        if ext.shape[0] % 2 == 0:
            self._mask[-1] = 0.0  # Nyquist bin under noninteger rFFT shifts

        # Use PySINDy to construct features: fit once, then transform for each tau.
        self.library_ = (self.feature_library if self.feature_library is not None else
                         ps.PolynomialLibrary(degree=self.degree, include_bias=True))
        reference = (np.hstack([xs[self.valid_indices_], xs[self.valid_indices_]])
                     if self.poly_on == "both" else xs[self.valid_indices_])
        self.library_.fit(reference)
        if not hasattr(self.library_, "powers_"):
            raise TypeError("Analytic gradients currently require PySINDy PolynomialLibrary")
        self._powers = np.asarray(self.library_.powers_, dtype=int)
        names = self._names + ([f"{name}_delay" for name in self._names]
                               if self.poly_on == "both" else [])
        self.feature_names_ = list(self.library_.get_feature_names(names))
        if self.poly_on == "current":
            self.feature_names_.extend([self.delay_feature_name]
                                       if self.delay_function is not None else
                                       [f"{name}_delay" for name in self._names])
        elif self.delay_function is not None:
            self.feature_names_.append(self.delay_feature_name)

        rng = np.random.default_rng(self.random_state)
        starts = []
        if self.tau0 is not None:
            starts.append(float(self.tau0))
        start_lo = lo if lo is not None else self.dt_
        start_hi = hi if hi is not None else self.max_delay_
        starts.extend(rng.uniform(start_lo, start_hi,
                                  self.n_starts - len(starts)).tolist())
        self.candidates_ = []
        for start in starts:
            tau, dense_history = self._adam(start, support=None)
            support = None
            for _ in range(self.n_alternations):
                A, _ = self._design(tau)
                new_support = self._select_support(A)
                tau, _ = self._adam(tau, support=new_support)
                if support is not None and np.array_equal(support, new_support):
                    support = new_support
                    break
                support = new_support
            if support is None:
                support = np.ones((d, len(self.feature_names_)), dtype=bool)
            loss, _, coef = self._profile(tau, support)
            k = int(np.sum(support))
            n_obs = self.y_.size
            # BIC is an approximate iid-regression heuristic, not a DDE likelihood.
            score = (n_obs*np.log(max(loss, 1e-30)) + (k+1)*np.log(n_obs)
                     if self.selection == "bic" else loss)
            self.candidates_.append({"initial_tau":start, "tau":tau,
                                     "mse":loss, "bic":score if self.selection == "bic" else np.nan,
                                     "support":support.copy(), "coefficients":coef.copy(),
                                     "dense_trace":dense_history})
        best = min(self.candidates_, key=lambda item:
                   item["bic"] if self.selection == "bic" else item["mse"])
        self.delay_ = float(best["tau"])
        self.coef_ = best["coefficients"]  # shape (n_states, n_features)
        self.support_ = best["support"]
        self.mse_ = float(best["mse"])
        self.n_features_in_ = d
        self.n_output_features_ = len(self.feature_names_)
        return self

    def _delayed(self, tau):
        # Keep the trained FFT and crop fixed to ensure a true profile gradient.
        phase = (self._spectrum * self._mask[:, None] *
                 np.exp(-1j*self._omega*tau)[:, None])
        z = np.fft.irfft(phase, n=self._extended_length, axis=0)
        dz = np.fft.irfft(-1j*self._omega[:, None]*phase,
                          n=self._extended_length, axis=0)
        return z[:len(self.x_smoothed_)][self.valid_indices_], \
               dz[:len(self.x_smoothed_)][self.valid_indices_]

    def _design(self, tau):
        z, dz = self._delayed(tau)
        x = self.x_smoothed_[self.valid_indices_]
        V = np.hstack([x, z]) if self.poly_on == "both" else x
        A = np.asarray(self.library_.transform(V), dtype=float)
        DA = np.zeros_like(A)
        # Analytical chain rule, including x*z, z^2, and higher-order monomials.
        if self.poly_on == "both":
            for j, power in enumerate(self._powers):
                for l in range(self.n_states_):
                    pos = self.n_states_ + l
                    exponent = int(power[pos])
                    if exponent:
                        reduced = power.copy()
                        reduced[pos] -= 1
                        DA[:, j] += (exponent * np.prod(V**reduced, axis=1)
                                     * dz[:, l])
        if self.poly_on == "current":
            if self.delay_function is None:
                extra, dextra = z, dz
            else:
                extra = np.asarray(self.delay_function(z[:, 0]))[:, None]
                dextra = (np.asarray(self.delay_derivative(z[:, 0]))
                          * dz[:, 0])[:, None]
            A = np.hstack([A, extra])
            DA = np.hstack([DA, dextra])
        elif self.delay_function is not None:
            h = np.asarray(self.delay_function(z[:, 0]))[:, None]
            dh = (np.asarray(self.delay_derivative(z[:, 0]))*dz[:, 0])[:, None]
            A, DA = np.hstack([A, h]), np.hstack([DA, dh])
        return A, DA

    def _profile(self, tau, support=None):
        A, DA = self._design(tau)
        d, p = self.n_states_, A.shape[1]
        if support is None:
            support = np.ones((d, p), dtype=bool)
        coef = np.zeros((d, p))
        for target in range(d):
            active = support[target]
            if np.any(active):
                coef[target, active] = np.linalg.lstsq(
                    A[:, active], self.y_[:, target], rcond=None)[0]
        residual = A @ coef.T - self.y_
        loss = np.mean(residual**2)
        grad = 2 * np.mean(residual * (DA @ coef.T))
        return float(loss), float(grad), coef

    def _select_support(self, A):
        # Relative threshold in unit-norm feature/response coordinates.
        # PySINDy performs the actual iterative STLSQ selection.
        scale = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
        ys = np.maximum(np.linalg.norm(self.y_, axis=0), 1e-12)
        An, Yn = A/scale, self.y_/ys
        dense = np.linalg.lstsq(An, Yn, rcond=None)[0]
        cutoff = self.threshold * np.max(np.abs(dense))
        if self.optimizer is None:
            opt = ps.STLSQ(threshold=float(cutoff), alpha=0.0,
                           max_iter=20, normalize_columns=False, unbias=True)
        else:
            # Must be a fresh PySINDy optimizer for each support update.
            from sklearn.base import clone
            opt = clone(self.optimizer)
        opt.fit(An, Yn)
        sparse_coef = np.atleast_2d(np.asarray(opt.coef_, dtype=float))
        active = np.abs(sparse_coef) > 1e-12
        if active.shape != (self.n_states_, A.shape[1]):
            raise RuntimeError(f"Unexpected STLSQ coefficients shape {active.shape}")
        for j in range(self.n_states_):
            if not active[j].any():
                active[j, int(np.argmax(np.abs(dense[:, j])))] = True
        return active

    def _adam(self, tau, support):
        m = v = 0.0
        trace = [float(tau)]
        for k in range(1, self.adam_iters + 1):
            _, g, _ = self._profile(tau, support)
            if not np.isfinite(g):
                raise FloatingPointError("Nonfinite delay gradient")
            m = 0.9*m + 0.1*g
            v = 0.999*v + 0.001*g*g
            tau -= self.adam_lr * (m/(1-0.9**k)) / (
                np.sqrt(v/(1-0.999**k)) + 1e-12)
            # Upper bound is physically dictated by finite available history.
            if self.delay_bounds is None:
                tau = float(np.clip(tau, self.dt_, self.max_delay_))
            else:
                tau = float(np.clip(tau, *self.delay_bounds))
            trace.append(tau)
        return float(tau), trace

    def loss_profile(self, tau, support="sparse"):
        """Return (MSE, analytic dMSE/dtau) at a candidate delay.

        support='dense' uses all terms; 'sparse' uses the fitted support.
        An explicit Boolean support array is also accepted.
        """
        if isinstance(support, str):
            if support == "dense":
                active = None
            elif support == "sparse":
                active = self.support_
            else:
                raise ValueError("support must be 'dense', 'sparse', or a mask")
        else:
            active = np.asarray(support, dtype=bool)
        loss, grad, _ = self._profile(float(tau), active)
        return loss, grad

    def get_feature_names(self):
        return list(self.feature_names_)

    def coefficients(self):
        return self.coef_.copy()

    def equations(self, precision=5):
        out = []
        for j in range(self.n_states_):
            terms = [f"({c:.{precision}g})*{name}"
                     for c, name in zip(self.coef_[j], self.feature_names_)
                     if abs(c) > 1e-12]
            out.append(" + ".join(terms) if terms else "0")
        return out

    def print(self, precision=5):
        print(f"Estimated delay: {self.delay_:.6g}")
        for name, eq in zip(self._names, self.equations(precision)):
            print(f"({name})' = {eq}")

    def predict(self, x=None, t=None):
        """Offline estimated drift at the valid interior samples.

        Returns shape (n_valid, d) or (n_valid,) for scalar input.
        This is NOT recursive/causal DDE trajectory simulation.
        For out-of-sample evaluation, pass a sufficiently long, uniformly
        sampled new trajectory and its dt; no fitting occurs.
        """
        if x is None:
            A, _ = self._design(self.delay_)
        else:
            # For new data, preserve trained feature library and call it on a
            # new full signal using a temporary override of the cached FFT.
            raw = self._to_2d(x)
            dt = self._get_dt(self.dt_ if t is None else t, len(raw))
            if not np.isclose(dt, self.dt_):
                raise ValueError("Prediction dt must match fitted dt")
            if raw.shape[1] != self.n_states_:
                raise ValueError("Prediction state dimension mismatch")
            if len(raw) <= 2*self.buffer_ + 2:
                raise ValueError("Prediction record too short for boundary buffer")
            xs = savgol_filter(raw, self.sg_window, self.sg_order, axis=0)
            ext = np.concatenate([xs, xs[-2:0:-1]], axis=0)
            old = (self.x_smoothed_, self.valid_indices_, self._spectrum,
                   self._extended_length, self._omega, self._mask)
            try:
                self.x_smoothed_ = xs
                self.valid_indices_ = np.arange(self.buffer_, len(raw)-self.buffer_)
                self._extended_length = len(ext)
                self._spectrum = np.fft.rfft(ext, axis=0)
                self._omega = 2*np.pi*np.fft.rfftfreq(len(ext), self.dt_)
                self._mask = np.ones_like(self._omega)
                if self.cutoff_period is not None:
                    self._mask = (self._omega <= 2*np.pi/self.cutoff_period).astype(float)
                if len(ext) % 2 == 0:
                    self._mask[-1] = 0
                A, _ = self._design(self.delay_)
            finally:
                (self.x_smoothed_, self.valid_indices_, self._spectrum,
                 self._extended_length, self._omega, self._mask) = old
        predicted = A @ self.coef_.T
        return predicted[:, 0] if self.n_states_ == 1 else predicted

    def score(self, x=None, t=None, x_dot=None):
        """Coefficient of determination (R^2) for estimated drift on valid times.

        Like predict(), this is an offline regression score, not a DDE
        trajectory forecasting score.
        """
        if x is None:
            truth = self.y_
            prediction = self.predict()
        else:
            raw = self._to_2d(x)
            if x_dot is None:
                truth = savgol_filter(raw, self.sg_window, self.sg_order,
                                      deriv=1, delta=self.dt_, axis=0)
            else:
                truth = self._to_2d(x_dot)
                if truth.shape != raw.shape:
                    raise ValueError("x_dot shape must match x")
            truth = truth[self.buffer_:len(raw)-self.buffer_]
            prediction = self.predict(x, t=t)
        prediction = self._to_2d(prediction)
        rss = np.sum((truth - prediction)**2)
        tss = np.sum((truth - truth.mean(axis=0))**2)
        return float(1 - rss / tss) if tss > 0 else np.nan
