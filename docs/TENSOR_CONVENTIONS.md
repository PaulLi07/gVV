# Lorentz tensor and angular-momentum conventions

This document fixes the conventions used by the reusable tensor building
blocks. It also records the normalization boundary followed by registered and
future process Waves. The framework APIs described here are independent of
the GVV final state; a complete production-and-decay Wave remains
process-specific.

The twelve registered spin-two GVV Waves consume this contract. The nominal
model activates the `LS=02` `U1/U2/U3` triplets for `f2(1565)` and `f2(1810)`;
the `LS=20,22,42` Waves remain registered but inactive unless selected by
`model.json`.

## 1. Stored indices and metric

`FV` stores contravariant components in the order

```text
[E, px, py, pz]
```

and `tensor` stores $A^{\mu\nu}$. The metric and Levi-Civita conventions are

\[
g_{\mu\nu}=\operatorname{diag}(1,-1,-1,-1),
\qquad \epsilon^{0123}=+1.
\]

Consequently,

\[
A^{\mu\nu}v_\nu
=\sum_\nu g_{\nu\nu}A^{\mu\nu}v^\nu,
\]

and every tensor-tensor contraction must explicitly lower the contracted
indices. `framework/tensors/TensorContraction.cuh` provides named operations
for that purpose:

| Function | Mathematical operation |
|---|---|
| `contract_second_index(A, v)` | $A^{\mu\nu}v_\nu$ |
| `contract_second_indices(A, B)` | $C^{\mu\nu}=A^{\mu\alpha}B^\nu{}_{\alpha}$ |
| `double_contract(A, B)` | $A^{\mu\nu}B_{\mu\nu}$ |
| `lorentz_trace(A)` | $A^\mu{}_{\mu}$ |
| `symmetrize(A)` | $A^{(\mu\nu)}$ |

New code should use these named functions instead of inventing a
`tensor * tensor` operator whose index placement would be unclear.

## 2. Transverse metric and spin-two projection

For a timelike parent momentum $K$, the transverse metric is

\[
G^{\mu\nu}(K)
=g^{\mu\nu}-\frac{K^\mu K^\nu}{K^2}.
\]

`transverse_projector(K)` returns this rank-two object. The spin-two
polarization projector is

\[
P^{(2)\mu\nu}{}_{\rho\sigma}(K)
=\frac12\left(
G^\mu{}_{\rho}G^\nu{}_{\sigma}
+G^\mu{}_{\sigma}G^\nu{}_{\rho}
\right)
-\frac13G^{\mu\nu}G_{\rho\sigma}.
\]

The framework does not allocate a rank-four projector. Instead,
`spin2_project(K, A)` applies it directly to a rank-two source. The returned
tensor is symmetric, transverse to $K$ in both indices, traceless, and
idempotent under a second application of the same projector.

## 3. Bare covariant orbital tensors

The project relative-momentum convention remains

\[
r=k_1-k_2,
\qquad
a^\mu=G^\mu{}_{\nu}(K)r^\nu.
\]

The framework orbital helpers return bare symmetric-traceless geometry. They
do not contain a Blatt-Weisskopf factor or an $LS$ normalization:

\[
t^{(1)\mu}_{\rm bare}=a^\mu,
\]

\[
t^{(2)\mu\nu}_{\rm bare}
=a^\mu a^\nu-\frac13a^2G^{\mu\nu},
\]

and

\[
\begin{aligned}
t^{(4)\mu\nu\lambda\tau}_{\rm bare}
={}&a^\mu a^\nu a^\lambda a^\tau\\
&-\frac{a^2}{7}\sum_{6}G^{\mu\nu}a^\lambda a^\tau
+\frac{(a^2)^2}{35}\sum_{3}G^{\mu\nu}G^{\lambda\tau}.
\end{aligned}
\]

The sums denote all six distinct one-trace permutations and all three
distinct two-trace pairings.

A general rank-four container would cost 256 doubles per CUDA thread and is
not needed by the planned amplitude. `contract_orbital_gwave(K, r, S)`
therefore evaluates the useful contraction directly:

\[
D^{\mu\nu}
=t^{(4)\mu\nu\lambda\tau}_{\rm bare}S_{\lambda\tau}.
\]

The function first keeps only the spin-two part of `S`, which cannot change
the contraction. Defining

\[
v^\mu=S^{\mu\nu}a_\nu,
\qquad s=a_\mu S^{\mu\nu}a_\nu,
\]

the implemented reduced form is

\[
\begin{aligned}
D^{\mu\nu}={}&a^\mu a^\nu s
-\frac{a^2}{7}\left[
G^{\mu\nu}s+2(a^\mu v^\nu+v^\mu a^\nu)
\right]
+\frac{2(a^2)^2}{35}S^{\mu\nu}.
\end{aligned}
\]

This result is already spin two. An additional outer $P^{(2)}$ is
mathematically redundant and must not be added around it.

## 4. Barrier-factor separation

`blatt_weisskopf(q, L, R)` supports consecutive $L=0,1,2,3,4$, with $q$
in GeV, $R$ in fm, and

\[
q_0=\frac{\hbar c}{R},
\qquad \hbar c=0.197321\ {\rm GeV\,fm}.
\]

The normalization is the existing CTPWA convention. In particular,

\[
B_4(q)=\sqrt{\frac{12746}{
q^8+10q_0^2q^6+135q_0^4q^4+1575q_0^6q^2+11025q_0^8}}.
\]

At $q=q_0$, $q^L B_L(q)=1$. A process Wave multiplies the appropriate
barrier factor exactly once. Orbital helpers never multiply it implicitly.

## 5. Normalized-CG convention for registered and future Waves

The generic APIs above deliberately preserve the existing bare-STF
normalization. The registered high-spin GVV Waves, and any future high-spin
Waves, use the following single normalized-CG convention in their process
layer.

In the parent rest frame, use the Condon-Shortley spherical vectors

\[
\boldsymbol e_{+1}=-\frac{\hat x+i\hat y}{\sqrt2},
\qquad
\boldsymbol e_0=\hat z,
\qquad
\boldsymbol e_{-1}=\frac{\hat x-i\hat y}{\sqrt2},
\]

and Racah-normalized harmonics

\[
C_{Lm}(\hat r)=\sqrt{\frac{4\pi}{2L+1}}Y_{Lm}(\hat r),
\qquad C_{L0}(\hat z)=1.
\]

The normalized orbital tensor is

\[
O_L^{\rm norm}=n_LO_L^{\rm bare},
\qquad
n_L=\sqrt{\frac{(2L-1)!!}{L!}},
\]

so that

\[
n_2=\sqrt{\frac32},
\qquad
n_4=\sqrt{\frac{35}{8}}.
\]

With unit-normalized Cartesian polarization tensors (E^{(L)}_m), this fixes

\[
E^{(L)*}_m:O_L^{\rm norm}
=|\boldsymbol r|^L C_{Lm}(\hat r),
\]

so the convention is explicitly tied to Racah (C_{Lm}), not to the
unit-integral (Y_{Lm}).

For the spin-zero coupling of the two vector currents, use the positive
Minkowski convention

\[
S_{00}^{\rm norm}=\frac{\Omega_1\mathbin{\cdot}\Omega_2}{\sqrt3}
\qquad \text{for the project }(+---)\text{ metric}.
\]

For the four implemented $X(2^{++})\to\omega\omega$ decay tensors, the
normalization is factored as follows. The last column is the total coefficient
that would multiply a formula written directly with the framework bare STF
objects.

| $LS$ | Orbital normalization | Spin normalization | $L\otimes S\to2$ factor | Total bare-form coefficient |
|---:|---:|---:|---:|---:|
| 02 | $1$ | $1$ | $1$ | $1$ |
| 20 | $\sqrt{3/2}$ | $1/\sqrt3$ | $1$ | $1/\sqrt2$ |
| 22 | $\sqrt{3/2}$ | $1$ | $\sqrt{12/7}$ | $\sqrt{18/7}$ |
| 42 | $\sqrt{35/8}$ | $1$ | $\sqrt5/3$ | $5\sqrt{14}/12$ |

The positive $D_{22}$ sign in this table follows the project's Minkowski contraction
`contract_second_indices`; replacing it with ordinary Euclidean matrix
multiplication would introduce the wrong sign. These constants normalize the
$X\to VV$ $LS$ tensors only. They do not normalize the independent
\(\psi\to\gamma X\) production covariants.

The standalone spin-two Wave files keep the orbital, spin, and coupling factors
visibly separate rather than hiding the last-column number in a generic
framework projector.

## 6. Verification contract

`tests/test_tensor_building_blocks.cu` compares the efficient APIs with
independent explicit-index references and checks:

- metric signs in every named contraction;
- the explicit spin-two projector, symmetry, transversality, trace removal,
  and idempotence;
- the reduced G-wave contraction against a test-only $4^4$ construction;
- the absence of any need for an outer spin-two projector;
- $r\to-r$ evenness, $r^4$ scaling, and two analytic rest-frame probes.

`tests/test_dynamics.cu` independently checks $B_3$ and $B_4$
polynomials and their normalization. The tensor regression is a CUDA runtime
test and is therefore executed only through the project's GPU-test Slurm
workflow, never directly on a login node.

`tests/test_gvv_wave_numerics.cu` evaluates every registered complete Wave,
checks finite nonzero Gram-matrix diagonals, Bose symmetry, rotation invariance,
and direct-versus-registry agreement for the active `LS=02` tensor Waves. The
process-neutral tensor test continues to validate the reusable contractions
without duplicating process-owned normalized-CG constants.
