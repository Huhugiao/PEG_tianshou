# Three-Party Differential Game Theory Applied to Missile Guidance Problem

## Nomenclature

| Symbol                   | Definition                                                                 |
|--------------------------|---------------------------------------------------------------------------|
| \(x_i = (x_i  y_i  z_i)^T\) | Position vector of vehicle \(i\) in fixed axis.                           |
| \(u_i = (u_i  v_i  w_i)^T\) | Velocity vector of vehicle \(i\) in fixed axis.                           |
| \(a_i = (a_{x_i}  a_{y_i}  a_{z_i})\) | Acceleration vector of vehicle \(i\) in fixed axis.                       |
| \((a_1^e, a_3^e)\)       | Evasion acceleration commands by target 1 (against attacker 3) and by attacker 3 (against defender 2). |
| \((a_2^p, a_3^p)\)       | Pursuit acceleration commands by defender 2 (against attacker 3) and by attacker 3 (against target 1). |
| \(x_{ij} = x_i - x_j\)   | Relative position vector of vehicle \(i\) w.r.t. vehicle \(j\).          |
| \(u_{ij} = u_i - u_j\)   | Relative velocity vector of vehicle \(i\) w.r.t. vehicle \(j\).          |
| \(y_{-31} = (x_{-31}  u_{-31})^T\) | Relative state vector of attacker 3 w.r.t. target 1.                     |
| \(y_{-23} = (x_{23}  u_{23})^T\) | Relative state vector of defender 2 w.r.t. attacker 3.                   |
| \(F\)                    | State coefficient matrix.                                                |
| \(G\)                    | Input coefficient matrix.                                                |
| \(J_i(\cdots)\)          | Scalar quadratic performance index (PI).                                 |
| \(P_i\)                  | Symmetric positive definite matrix solution to matrix Riccati differential equation. |
| \(Q_i\)                  | Positive semi-definite matrix of PI weightings on current states.        |
| \(\{R_1^e, R_2^p, R_3^p, R_3^e\}\) | Positive-definite matrices of PI weightings on (control) inputs for target 1, defender 2, attacker 3. |
| \(S_i\)                  | Positive semi-definite matrix of PI weightings on final states.          |
| \(H_{i,j}(\cdots)\)      | Hamiltonian.                                                             |
| \(\lambda_i\)            | Euler-Lagrange operators used in a Hamiltonian.                          |

## Abbreviations
*   **3-D:** Three dimension
*   **4-DOF:** Four degrees of freedom
*   **AI:** Artificial intelligence
*   **LQPI:** Linear system quadratic performance index
*   **MD:** Miss distance
*   **MRDE:** Matrix Riccati differential equation
*   **PI:** Performance index
*   **VRDE:** Vector Riccati differential equation

## 4.1 Introduction
Reported research[1–9] on applying differential game theory to missile guidance has concentrated on two-party engagements (attacker/pursuer vs. evader/target). This chapter extends the approach to a **three-party engagement scenario**:
1.  **Primary Target (e.g., Aircraft, \(j=1\)):** Aware of being attacked, fires a defending missile and performs evasive maneuvers.
2.  **Attacking Missile (Attacker, \(i=3\)):** Aims to intercept the primary target while evading the defending missile (dual role).
3.  **Defending Missile (Defender, \(i=2\)):** Aims solely to intercept the attacking missile.

Participants have mutually conflicting objectives, forming a **three-party differential game**. Previous work used the Linear Quadratic Performance Index (LQPI) approach for two-party games, yielding analytical solutions for guidance commands as functions of LQPI weightings and time-to-go (\(T = t_f - t\)). Chapter 3 generalized the theoretical basis for two-party games. This chapter:
1.  Extends the framework to a three-party scenario.
2.  Develops a 3-D engagement kinematics model.
3.  Obtains a solution via the matrix Riccati equation (MRDE).
4.  Discusses incorporating rule-based AI inputs to enhance evasion/pursuit strategies.

**Chapter Structure:**
*   **Section 4.2:** Derives the engagement kinematics model.
*   **Section 4.3:** Sets up the three-party game optimization problem using LQPI, leading to the MRDE and feedback guidance laws.
*   **Section 4.4:** Solves the MRDE and VRDE.
*   **Section 4.5:** Discusses results and conclusions.

## 4.2 Engagement Kinematics Model
Kinematics variables are defined in a fixed axis system (Figure 4.2.1). Equations for vehicle \(i\) (\(i = 1, 2, 3\)):
\[\frac{d}{dt} x_i = u_i \quad \frac{d}{dt} y_i = v_i \quad \frac{d}{dt} z_i = w_i \tag{4.2.1-4.2.3}\]
\[\frac{d}{dt} u_i = a_{x_i} \quad \frac{d}{dt} v_i = a_{y_i} \quad \frac{d}{dt} w_i = a_{z_i} \tag{4.2.4-4.2.6}\]
Where:
*   \(\dot{x}_i = (x_i  y_i  z_i)^T\): Position vector.
*   \(u_i = (u_i  v_i  w_i)^T\): Velocity vector.
*   \(a_i = (a_{x_i}  a_{y_i}  a_{z_i})\): Acceleration vector.

**State Space Form:**
\[\frac{d}{dt} \begin{bmatrix} \dot{x}_i \\ \vdots \\ u_i \end{bmatrix} = \begin{bmatrix} 0 & \cdots & 1 \\ 0 & \cdots & 0 \\ 0 & \cdots & 0 \end{bmatrix} \begin{bmatrix} \dot{x}_i \\ \vdots \\ u_i \end{bmatrix} + \begin{bmatrix} 0 \\ \vdots \\ 1 \end{bmatrix} [a_i] \tag{4.2.7}\]

**Relative Kinematics (Vehicle \(j\) w.r.t. Vehicle \(i\)):**
\[\frac{d}{dt} \begin{bmatrix} \dot{x}_{-ij} \\ \vdots \\ u_{-ji} \end{bmatrix} = \begin{bmatrix} 0 & \cdots & 1 \\ 0 & \cdots & 0 \\ 0 & \cdots & 0 \end{bmatrix} \begin{bmatrix} \dot{x}_{-ij} \\ \vdots \\ u_{-ji} \end{bmatrix} + \begin{bmatrix} 0 \\ \vdots \\ 1 \end{bmatrix} a_i - \begin{bmatrix} 0 \\ \vdots \\ 1 \end{bmatrix} a_j \tag{4.2.8}\]
Where:
*   \(\dot{x}_{-ij} = x_i - x_j\): (3×1) Relative position vector.
*   \(u_{-ji} = u_i - u_j\): (3×1) Relative velocity vector.
*   \(I\): (3×3) Identity matrix; \(j \neq i\).

### 4.2.1 Three-Party Engagement Scenario
*   **Target 1 (\(j=1\)) vs. Attacker 3 (\(i=3\)):**
    *   States: \((x_{31}  u_{31})\)
    *   Inputs: \((a_3, a_1)\)
        *   \(a_1 = a_1^e\) (evasion by target 1)
        *   \(a_3\) includes \(a_3^p\) (pursuit by attacker 3)
*   **Defender 2 (\(i=2\)) vs. Attacker 3 (\(j=3\)):**
    *   States: \((x_{23}  u_{23})\)
    *   Inputs: \((a_2, a_3)\)
        *   \(a_3\) includes \(a_3^e\) (evasion by attacker 3)
        *   \(a_2 = a_2^p\) (pursuit by defender 2)

**Acceleration Command Decomposition:**
*   \(a_1 = a_1^e + a_1^d\): Evasion + Disturbance (Target 1)
*   \(a_2 = a_2^p\): Pursuit (Defender 2)
*   \(a_3 = a_3^p + a_3^e + a_3^d\): Pursuit (vs T1) + Evasion (vs D2) + Disturbance (Attacker 3) **(4.2.9)**
Where:
*   \((a_1^e, a_3^e)\): Evasion acceleration commands (3x1-vectors).
*   \((a_2^p, a_3^p)\): Pursuit acceleration commands (3x1-vectors).
*   \((a_1^d, a_3^d)\): Disturbance acceleration commands (3x1-vectors) for additional maneuvers.

**Kinematics Models:**
1.  **Attacker 3 vs. Target 1 (\(0 \leq t \leq t_{f_1}\)):**
    \[\frac{d}{dt} \begin{bmatrix} \mathbf{x}_{31} \\ \vdots \\ \mathbf{u}_{31} \end{bmatrix} = \begin{bmatrix} 0 & : & I \\ 0 & : & 0 \end{bmatrix} \begin{bmatrix} \mathbf{x}_{31} \\ \vdots \\ \mathbf{u}_{31} \end{bmatrix} + \begin{bmatrix} 0 \\ \vdots \\ I \end{bmatrix} \left( \mathbf{a}^p_3 + \mathbf{a}^e_3 + \mathbf{a}^d_3 \right) - \begin{bmatrix} 0 \\ \vdots \\ I \end{bmatrix} \left( \mathbf{a}^e_1 + \mathbf{a}^d_1 \right) \tag{4.2.10}\]
    \[\frac{d}{dt}\mathbf{y}_{31} = [F]\mathbf{y}_{31} + [G]\left( \mathbf{a}^p_3 - \mathbf{a}^e_1 \right) - [G]\left( \mathbf{a}^d_1 - \mathbf{a}^e_3 - \mathbf{a}^d_3 \right) \tag{4.2.11}\]
    *   Inputs \((\mathbf{a}^p_3 - \mathbf{a}^e_1)\): Determine strategies (lumped).
    *   Inputs \((\mathbf{a}^d_1 - \mathbf{a}^e_3 - \mathbf{a}^d_3)\): "Additional" inputs affecting VRDE.
2.  **Defender 2 vs. Attacker 3 (\(0 \leq t \leq t_{f_2}\)):**
    \[\frac{d}{dt} \begin{bmatrix} x_{23} \\ \vdots \\ x_{23} \end{bmatrix} = \begin{bmatrix} 0 & : & I \\ 0 & : & 0 \end{bmatrix} \begin{bmatrix} x_{23} \\ \vdots \\ x_{23} \end{bmatrix} + \begin{bmatrix} 0 \\ \vdots \\ I \end{bmatrix} \begin{pmatrix} a_p^p \\ \vdots \\ a_q^p \end{pmatrix} - \begin{bmatrix} 0 \\ \vdots \\ I \end{bmatrix} \begin{pmatrix} a_p^p + a_q^e + a_q^d \\ a_q^p + a_q^q \end{pmatrix} \tag{4.2.12}\]
    \[\frac{d}{dt} y_{23} = [F] y_{23} + [G] \begin{pmatrix} a_p^p - a_q^e \\ -2a_q^e \end{pmatrix} - [G] \begin{pmatrix} a_q^d + a_p^p \\ -3a_q^d \end{pmatrix} \tag{4.2.13}\]
    *   Inputs \((a_p^p - a_q^e)\): Determine strategies (lumped).
    *   Inputs \((a_q^d + a_p^p)\): "Additional" inputs.

**Definitions:**
*   \(y_{31} = \begin{pmatrix} x_{31} & u_{31} \end{pmatrix}^T\): (6×1) Relative state vector (Interceptor 3 / Target 1).
*   \(y_{23} = \begin{pmatrix} x_{23} & u_{23} \end{pmatrix}^T\): (6×1) Relative state vector (Defender 2 / Attacker 3).
*   \([F]\): (6×6) State coefficient matrix.
*   \([G]\): (6×3) Input coefficient matrix.

## 4.3 Three-Party Differential Game Problem and Solution
**Problem Statement:** Given dynamics (4.2.11) and (4.2.13) with initial states \(y_{31}(t_0) = y_{31}(0)\), \(y_{23}(t_0) = y_{23}(0)\), and scalar quadratic PIs:
\[J_1(\cdots) = \frac{1}{2} \|y_{31}(t_f)\|_{S_1}^2 + \frac{1}{2} \int_{t_0}^{t_f} \left[ \|y_{31}\|_{Q_1}^2 + \|a_{3}^p\|_{R_3^p}^2 - \|a_{1}^e\|_{R_1^e}^2 \right] dt \tag{4.3.1}\]
\[J_2(\cdots) = \frac{1}{2} \|y_{23}(t_f)\|_{S_2}^2 + \frac{1}{2} \int_{t_0}^{t_f} \left[ \|y_{23}\|_{Q_2}^2 + \|a_{2}^p\|_{R_2^p}^2 - \|a_{3}^e\|_{R_3^e}^2 \right] dt \tag{4.3.2}\]
Where:
*   \([S_1, S_2]\): (6×6) Positive semi-definite final-state weighting matrices.
*   \([Q_1, Q_2]\): (6×6) Positive semi-definite current-state weighting matrices.
*   \([R_1^e, R_2^p, R_3^p, R_3^e]\): (3×3) Positive-definite control input weighting matrices ("soft constraints").

**Specific Case:** \([Q_i] = [0]\); \([S_i] = \text{diag}[s_1 \ s_2 \ s_3 \ 0 \ 0 \ 0]\), \(s_1 = s_2 = s_3 = s; \ i = 1, 2, 3\). Final terms become weighted miss distances:
\[\|y_{-31}(t_f)\|_{s_1} = s\|x_{-31}(t_f)\|, \quad \|y_{-23}(t_f)\|_{s_2} = s\|x_{-23}(t_f)\|\]

**Control Weighting Assumption:** \([R_1^e = r_1^e I; \ R_2^p = r_2^p I; \ R_3^p = r_3^p I; \ R_3^e = r_3^e I]\) (scalars \(r_1^e, r_2^p, r_3^p, r_3^e\)). **Requirement for Solution:** \(r_1^e > r_3^p\), \(r_3^e > r_2^p\) (Evasion weight > Pursuit weight).

**Objective:** Derive guidance commands \((a^e_1, a^p_2, a^p_3, a^e_3)\) optimizing \(J^*(...)\). The negative signs on evasion terms transform the max/min problem into a minimization problem:
\[J^*_1(...) = \min_{\begin{pmatrix} a^p_3, a^e_1 \end{pmatrix}} J_1; \quad J^*_2(...) = \min_{\begin{pmatrix} a^p_2, a^e_3 \end{pmatrix}} J_2(...) \tag{4.3.3}\]
\[\min_{\begin{pmatrix} a^p_3, a^p_2, a^e_1, a^e_3 \end{pmatrix}} [J_1(...) + J_2(...)] = \min_{\begin{pmatrix} a^p_3, a^e_1 \end{pmatrix}} J_1 + \min_{\begin{pmatrix} a^p_2, a^e_3 \end{pmatrix}} J_2(...) \tag{4.3.4}\]

**Assumptions:**
*   Engagement times \(t_{f_2}\) (D2 vs A3) and \(t_{f_1}\) (A3 vs T1) may differ.
*   All parties have access to full relative state information \(\{x_{ij}(t); \ \forall t_0 \leq t \leq t_f\}\).
*   \((a^p_3, a^p_2)\) minimize \(J_1(...)\), \(J_2(...)\) (Pursuit).
*   \((a^e_3, a^e_1)\) maximize \(J_1(...)\), \(J_2(...)\) (Evasion - achieved via negative sign & min. problem).

**Solution via LQPI/Hamiltonian:**
Hamiltonians \( H_1(...), H_2(...) \):
\[H_1(...) = \frac{1}{2} \left\{ \left\| a_{3}^{P} \right\|_{R_3^{P}}^{2} - \left\| a_{1}^{e} \right\|_{R_1^{e}}^{2} \right\} + \lambda_1^T \left\{ [F]_{Y_{-31}} + [G] \left( a_{3}^{P} - a_{1}^{e} \right) - [G] \left( a_{1}^{d} - a_{3}^{e} - a_{3}^{d} \right) \right\} \tag{4.3.5}\]
\[H_2(...) = \frac{1}{2} \left\{ \left\| a_{2}^{P} \right\|_{R_2^{P}}^{2} - \left\| a_{3}^{e} \right\|_{R_3^{e}}^{2} \right\} + \lambda_2^T \left\{ [F]_{Y_{-23}} + [G] \left( a_{2}^{P} - a_{3}^{e} \right) - [G] \left( a_{3}^{d} + a_{3}^{p} \right) \right\} \tag{4.3.6}\]

**Necessary Conditions for Optimality (\(\partial H / \partial a = 0\)):**
\[\frac{\partial H_1}{\partial a_{1}^{e}} = - \left[ R_1^{e} \right] a_{1}^{e} - [G]^T \lambda_1 = 0 \tag{4.3.7}\]
\[\frac{\partial H_1}{\partial a_{3}^{P}} = \left[ R_3^{P} \right] a_{3}^{P} + [G]^T \lambda_1 = 0 \tag{4.3.8}\]
\[\frac{\partial H_1}{\partial a_{3}^{e}} = - [G]^T \lambda_1 = 0 \tag{4.3.9}\]
\[\frac{\partial H_2}{\partial a_{2}^{P}} = \left[ R_2^{P} \right] a_{2}^{P} + [G]^T \lambda_2 = 0 \tag{4.3.10}\]
\[\frac{\partial H_2}{\partial a_{3}^{e}} = - \left[ R_3^{e} \right] a_{3}^{e} - [G]^T \lambda_2 = 0 \tag{4.3.11}\]
\[\frac{\partial H_2}{\partial a_{3}^{P}} = - [G]^T \lambda_2 = 0 \tag{4.3.12}\]
*Note:* Equations (4.3.9) and (4.3.12) do not yield optimum values for \(a_{3}^{e}\) and \(a_{3}^{P}\). Their optimums are defined by (4.3.11) and (4.3.8). In deriving the Riccati equations:
    *   \(a_{3}^{e}\) is treated as an additional input in the system for \(y_{31}\).
    *   \(a_{3}^{P}\) is treated as an additional input in the system for \(y_{23}\).

**Optimization Conditions:**
\[\frac{\partial H_1}{\partial y} = -\dot{\lambda}_1 = [F]^T \lambda_1 \tag{4.3.13}\]
\[\frac{\partial H_2}{\partial y} = -\dot{\lambda}_2 = [F]^T \lambda_2 \tag{4.3.14}\]

**Boundary Conditions & Riccati Solution Form:**
\(\lambda_{-1}(t_{f_1}) = [S_1]y_{-31}(t_{f_1})\), \(\lambda_{-2}(t_{f_2}) = [S_2]y_{-23}(t_{f_2})\). Assume:
\[\lambda_{-1} = [P_1]y_{-31} + \xi_{-1}, \quad \lambda_{-2} = [P_2]y_{-23} + \xi_{-2}\]
Solving (4.3.7, 4.3.8, 4.3.10, 4.3.11) gives:
\[\begin{aligned}
a_1^e &= - \left[ R_1^e \right]^{-1} \left[ G \right]^T \left( [P_1]y_{-31} + \xi_{-1} \right) \tag{4.3.15} \\
a_3^p &= - \left[ R_3^p \right]^{-1} \left[ G \right]^T \left( [P_1]y_{-31} + \xi_{-1} \right) \tag{4.3.16} \\
a_2^p &= - \left[ R_2^p \right]^{-1} \left[ G \right]^T \left( [P_2]y_{-23} + \xi_{-2} \right) \tag{4.3.17} \\
a_3^e &= - \left[ R_3^e \right]^{-1} \left[ G \right]^T \left( [P_2]y_{-23} + \xi_{-2} \right) \tag{4.3.18}
\end{aligned}\]
Where \([P_i]\): (6×6) Riccati matrix, \(\xi_i\): (6×1) Riccati vector (\(i = 1, 2\)).

**Riccati Differential Equations (Derived by substitution & manipulation):**
1.  **Matrix Riccati DE (MRDE):**
    \[[\dot{P}_1] + [P_1][F] + [F]^T[P_1] - [P_1][G][R_{31}]^{-1}[G]^T[P_1] = 0 \tag{4.3.19}\]
    \[[\dot{P}_2] + [P_2][F] + [F]^T[P_2] - [P_2][G][R_{23}]^{-1}[G]^T[P_2] = 0 \tag{4.3.21}\]
    Where: \([R_{31}]^{-1} = \left( \left[ R_3^p \right]^{-1} - \left[ R_1^e \right]^{-1} \right)\), \([R_{23}]^{-1} = \left( \left[ R_2^p \right]^{-1} - \left[ R_3^e \right]^{-1} \right)\)
2.  **Vector Riccati DE (VRDE):**
    \[\dot{\xi}_{-1} + \{ [F]^T - [P_1][G][R_{31}][G]^T \} \xi_{-1} - [P_1][G] \left( a_1^d - a_3^e - a_3^d \right) = 0 \tag{4.3.20}\]
    \[\dot{\xi}_{-2} + \{ [F]^T \xi_{-2} - [P_2][G][R_{23}][G]^T \} \xi_{-2} - [P_2][G] \left( a_3^d + a_3^p \right) = 0 \tag{4.3.22}\]

**Boundary Conditions:** \(P_1(t_{f_1}) = S_1\), \(P_2(t_{f_2}) = S_2\), \(\xi_{-1}(t_{f_1}) = 0\), \(\xi_{-2}(t_{f_2}) = 0\).

## 4.4 Solution of the Riccati Differential Equations
### 4.4.1 Solution of the Matrix Riccati Differential Equation (MRDE)
Consider case: \([Q_i] = [0]\); \([S_i] = \text{diag}[s_1 \ s_2 \ s_3 \ 0 \ 0 \ 0]\); \(s_1 = s_2 = s_3 = s; \ i = 1, 2\); \([R_{31}] = r_{31}I\), \([R_{23}] = r_{23}I\) (scalars). Analytical solution for \([P_i]\) as a function of time-to-go (\(T_i = t_{f_i} - t\)):
\[[P_i] = \begin{bmatrix}
p_{11i} & 0 & 0 & p_{14i} & 0 & 0 \\
0 & p_{22i} & 0 & 0 & p_{25i} & 0 \\
0 & 0 & p_{33i} & 0 & 0 & p_{36i} \\
p_{14i} & 0 & 0 & p_{44i} & 0 & 0 \\
0 & p_{25i} & 0 & 0 & p_{55i} & 0 \\
0 & 0 & p_{36i} & 0 & 0 & p_{66i}
\end{bmatrix} \tag{4.4.1}\]
Where (\(i=1\): Solution of (4.3.19), \(i=2\): Solution of (4.3.21)):
\[\begin{aligned}
p_{11i} &= p_{22i} = p_{33i} = \left[ \frac{3\gamma_i}{3\gamma_i + T_i^3} \right] \\
p_{14i} &= p_{25i} = p_{36i} = \left[ \frac{3\gamma_i T_i}{3\gamma_i + T_i^3} \right] \\
p_{44i} &= p_{55i} = p_{66i} = \left[ \frac{3\gamma_i T_i^2}{3\gamma_i + T_i^3} \right]
\end{aligned}\]
With:
*   \([R_1^e = r_1^e I; \ R_2^p = r_2^p I; \ R_3^p = r_3^p I; \ R_3^e = r_3^e I]\) (scalars \(r_1^e, r_2^p, r_3^p, r_3^e\)).
*   \(\gamma_1 = r_{31} = \frac{r_3^p r_1^e}{(r_1^e - r_3^p)}; \quad \gamma_2 = r_{23} = \frac{r_2^p r_3^e}{(r_3^e - r_2^p)}\)
*   \(T_i = (t_{f_i} - t)\) (Time-to-go).

**Relationships:**
\[p_{14i} = p_{25i} = p_{36i} = T_i p_{11i} = T_i p_{22i} = T_i p_{33i}\]
\[p_{44i} = p_{55i} = p_{66i} = T_i p_{14i} = T_i p_{25i} = T_i p_{36i}\]

**Existence Condition:** \(r_{31} > 0\), \(r_{23} > 0\) \(\implies\) \(r_1^e > r_3^p\), \(r_3^e > r_2^p\) (Evasion weight > Pursuit weight).

**State Feedback Gain Matrices:** From (4.3.15-4.3.18) & (4.4.1):
\[\left[ K_{1}^{e} \right] = \frac{1}{r_{1}^{e}} \left[ G \right]^{T} \left[ P_{1} \right] = \frac{1}{r_{1}^{e}} \left[ \frac{3 r_{31} T_{1}}{3 r_{31} + T_{1}^{3}} \right] \begin{bmatrix} 1 & 0 & 0 & T_{1} & 0 & 0 \\ 0 & 1 & 0 & 0 & T_{1} & 0 \\ 0 & 0 & 1 & 0 & 0 & T_{1} \end{bmatrix} \tag{4.4.2, 4.4.6}\]
\[\left[ K_{3}^{p} \right] = \frac{1}{r_{3}^{p}} \left[ G \right]^{T} \left[ P_{1} \right] = \frac{1}{r_{3}^{p}} \left[ \frac{3 r_{31} T_{1}}{3 r_{31} + T_{1}^{3}} \right] \begin{bmatrix} 1 & 0 & 0 & T_{1} & 0 & 0 \\ 0 & 1 & 0 & 0 & T_{1} & 0 \\ 0 & 0 & 1 & 0 & 0 & T_{1} \end{bmatrix} \tag{4.4.3, 4.4.6}\]
\[\left[ K_{2}^{p} \right] = \frac{1}{r_{2}^{p}} \left[ G \right]^{T} \left[ P_{2} \right] = \frac{1}{r_{2}^{p}} \left[ \frac{3 r_{23} T_{2}}{3 r_{23} + T_{2}^{3}} \right] \begin{bmatrix} 1 & 0 & 0 & T_{2} & 0 & 0 \\ 0 & 1 & 0 & 0 & T_{2} & 0 \\ 0 & 0 & 1 & 0 & 0 & T_{2} \end{bmatrix} \tag{4.4.4, 4.4.7}\]
\[\left[ K_{3}^{e} \right] = \frac{1}{r_{3}^{e}} \left[ G \right]^{T} \left[ P_{2} \right] = \frac{1}{r_{3}^{e}} \left[ \frac{3 r_{23} T_{2}}{3 r_{23} + T_{2}^{3}} \right] \begin{bmatrix} 1 & 0 & 0 & T_{2} & 0 & 0 \\ 0 & 1 & 0 & 0 & T_{2} & 0 \\ 0 & 0 & 1 & 0 & 0 & T_{2} \end{bmatrix} \tag{4.4.5, 4.4.7}\]

### 4.4.2 Solution of the Vector Riccati Differential Equation (VRDE)
Closed-form analytical solution for VRDE (4.3.20, 4.3.22) generally requires assumptions on the disturbance terms \((\mathbf{a}_{1}^{d} - \mathbf{a}_{3}^{d} - \mathbf{a}_{3}^{e})\) and \((\mathbf{a}_{3}^{d} + \mathbf{a}_{3}^{p})\).

**Solution Approach:**
1.  Substitute: \(T_{i} = t_{f_{i}} - t \rightarrow dT_{i} = -dt\); \(\xi_i(t) = \xi_i(t_{f_i} - T_i) = \eta_i(T_i) = \eta_i; i = 1, 2\).
2.  Define piecewise constant disturbance inputs over intervals \(T_{i,k} \geq T_i \geq T_{i,k+1}\):
    *   For \(i=1\) (VRDE 4.3.20): \(\mathbf{a}_{1}^{d}(t) = \alpha_{1}^{d}(T_{1})\), \(\mathbf{a}_{3}^{e}(t) = \alpha_{3}^{e}(T_{1})\), \(\mathbf{a}_{3}^{d}(t) = \alpha_{3}^{d}(T_{1})\) \(\implies\) Constants \(\alpha_{1,k}^{d}, \alpha_{3,k}^{e}, \alpha_{3,k}^{d}\).
    *   For \(i=2\) (VRDE 4.3.22): \(\mathbf{a}_{3}^{p}(t) = \beta_{3}^{p}(T_{2})\), \(\mathbf{a}_{3}^{d}(t) = \beta_{3}^{d}(T_{2})\) \(\implies\) Constants \(\beta_{3,k}^{p}, \beta_{3,k}^{d}\).

**Solution for VRDE (4.3.20) - \(i=1\):**
\[\begin{aligned}
\eta_{11} &= \left[ \frac{3r_{31}T_1^2}{3r_{31} + T_1^3} \right] \left( \alpha^d_{x_{1,k}} - \alpha^e_{x_{3,k}} - \alpha^d_{x_{3,k}} \right) \tag{4.4.8} \\
\eta_{21} &= \left[ \frac{3r_{31}T_1^2}{3r_{31} + T_1^3} \right] \left( \alpha^d_{y_{1,k}} - \alpha^e_{y_{3,k}} - \alpha^d_{y_{3,k}} \right) \tag{4.4.9} \\
\eta_{31} &= \left[ \frac{3r_{31}T_1^2}{3r_{31} + T_1^3} \right] \left( \alpha^d_{z_{1,k}} - \alpha^e_{z_{3,k}} - \alpha^d_{z_{3,k}} \right) \tag{4.4.10} \\
\eta_{41} &= \left[ \frac{3r_{31}T_1^3}{3r_{31} + T_1^3} \right] \left( \alpha^d_{x_{1,k}} - \alpha^e_{x_{3,k}} - \alpha^d_{x_{3,k}} \right) \tag{4.4.11} \\
\eta_{51} &= \left[ \frac{3r_{31}T_1^3}{3r_{31} + T_1^3} \right] \left( \alpha^d_{y_{1,k}} - \alpha^e_{y_{3,k}} - \alpha^d_{y_{3,k}} \right) \tag{4.4.12} \\
\eta_{61} &= \left[ \frac{3r_{31}T_1^3}{3r_{31} + T_1^3} \right] \left( \alpha^d_{z_{1,k}} - \alpha^e_{z_{3,k}} - \alpha^d_{z_{3,k}} \right) \tag{4.4.13}
\end{aligned}\]

**Solution for VRDE (4.3.22) - \(i=2\):**
\[\begin{aligned}
\eta_{12} &= \left[ \frac{3r_{23}T_2^2}{3r_{23} + T_2^3} \right] \left( \beta^p_{x_{3,k}} + \beta^d_{x_{3,k}} \right) \tag{4.4.14} \\
\eta_{22} &= \left[ \frac{3r_{23}T_2^2}{3r_{23} + T_2^3} \right] \left( \beta^p_{y_{3,k}} + \beta^d_{y_{3,k}} \right) \tag{4.4.15} \\
\eta_{32} &= \left[ \frac{3r_{23}T_2^2}{3r_{23} + T_2^3} \right] \left( \beta^p_{z_{3,k}} + \beta^d_{z_{3,k}} \right) \tag{4.4.16} \\
\eta_{42} &= \left[ \frac{3r_{23}T_2^3}{3r_{23} + T_2^3} \right] \left( \beta^p_{x_{3,k}} + \beta^d_{x_{3,k}} \right) \tag{4.4.17} \\
\eta_{52} &= \left[ \frac{3r_{23}T_2^3}{3r_{23} + T_2^3} \right] \left( \beta^p_{y_{3,k}} + \beta^d_{y_{3,k}} \right) \tag{4.4.18} \\
\eta_{62} &= \left[ \frac{3r_{23}T_2^3}{3r_{23} + T_2^3} \right] \left( \beta^p_{z_{3,k}} + \beta^d_{z_{3,k}} \right) \tag{4.4.19}
\end{aligned}\]

**Guidance Disturbance Inputs (From \(\xi_i = -[R]^{-1}[G]^T \xi_i\)):**
\[\underline{k}_1^e = -\frac{1}{r_1^e} \left[ \frac{3r_{31}T_1^3}{3r_{31} + T_1^3} \right] \begin{bmatrix} \alpha^d_{x_{1,k}} - \alpha^e_{x_{3,k}} - \alpha^d_{x_{3,k}} \\ \alpha^d_{y_{1,k}} - \alpha^e_{y_{3,k}} - \alpha^d_{y_{3,k}} \\ \alpha^d_{z_{1,k}} - \alpha^e_{z_{3,k}} - \alpha^d_{z_{3,k}} \end{bmatrix} \tag{4.4.20}\]
\[\underline{k}_3^p = -\frac{1}{r_3^p} \left[ \frac{3r_{31}T_1^3}{3r_{31} + T_1^3} \right] \begin{bmatrix} \alpha^d_{x_{1,k}} - \alpha^e_{x_{3,k}} - \alpha^d_{x_{3,k}} \\ \alpha^d_{y_{1,k}} - \alpha^e_{y_{3,k}} - \alpha^d_{y_{3,k}} \\ \alpha^d_{z_{1,k}} - \alpha^e_{z_{3,k}} - \alpha^d_{z_{3,k}} \end{bmatrix} \tag{4.4.21}\]
\[\underline{k}_2^p = -\frac{1}{r_2^p} \left[ \frac{3r_{23}T_2^3}{3r_{23} + T_2^3} \right] \begin{bmatrix} \beta^p_{x_{3,k}} + \beta^d_{x_{3,k}} \\ \beta^p_{y_{3,k}} + \beta^d_{y_{3,k}} \\ \beta^p_{z_{3,k}} + \beta^d_{z_{3,k}} \end{bmatrix} \tag{4.4.22}\]
\[\underline{k}_3^e = -\frac{1}{r_3^e} \left[ \frac{3r_{23}T_2^3}{3r_{23} + T_2^3} \right] \begin{bmatrix} \beta^p_{x_{3,k}} + \beta^d_{x_{3,k}} \\ \beta^p_{y_{3,k}} + \beta^d_{y_{3,k}} \\ \beta^p_{z_{3,k}} + \beta^d_{z_{3,k}} \end{bmatrix} \tag{4.4.23}\]

**Remarks (Implementation):**
1.  Guidance commands derived in fixed axis; applied in vehicle body axis (lateral accelerations). Longitudinal acceleration often assumed zero for missiles, making commands "sub-optimal" when applied. Transformation inclusion complicates Riccati equations.
2.  Autopilot lags not included in derivation (increases state model order). Used in simulation (Ch. 6) but not here.
3.  Meaningful scenarios require challenging initial conditions (unbiased engagement).
4.  Predetermined evasion maneuvers (step, sinusoidal, jinking) can be modeled via disturbance inputs.

### 4.4.3 Further Consideration of Performance Index (PI) Weightings
Rewrite PIs (4.3.1, 4.3.2) with specific weights (\(Q_1 = Q_2 = 0; S_1 = S_2 = I; R_1^e = r_1^eI; R_2^p = r_2^pI; R_3^p = r_3^pI; R_3^e = r_3^eI\)):
\[J_1(\cdots) = \frac{1}{2} \|y_{-31}(t_f)\|^2 + \frac{1}{2} \int_{t_0}^{t_f} \left[ r_3^P \|a_3^P\|^2 - r_1^e \|a_1^e\|^2 \right] dt \tag{4.4.24}\]
\[J_2(\cdots) = \frac{1}{2} \|y_{-23}(t_f)\|^2 + \frac{1}{2} \int_{t_0}^{t_f} \left[ r_2^P \|a_2^P\|^2 - r_3^e \|a_3^e\|^2 \right] dt \tag{4.4.25}\]
Set \(r_1^e = \alpha r_3^P; \quad r_3^e = \beta r_2^P; \quad \alpha, \beta > 1\):
\[J_1(\cdots) = \frac{1}{2} \|y_{-31}(t_f)\|^2 + \frac{1}{2} \int_{t_0}^{t_f} r_3^P \left( \|a_3^P\|^2 - \alpha \|a_1^e\|^2 \right) dt \tag{4.4.26}\]
\[J_2(\cdots) = \frac{1}{2} \|y_{-23}(t_f)\|^2 + \frac{1}{2} \int_{t_0}^{t_f} r_2^P \left( \|a_2^P\|^2 - \beta \|a_3^e\|^2 \right) dt \tag{4.4.27}\]
Assuming \(r_3^P, r_2^P \ll 1\) (high weighting on final miss):
*   **(a)** \(\alpha, \beta \gg 1\): Negligible evasion by T1 & A3; essentially pure intercept trajectories.
*   **(b)** \(\alpha, \beta > 1\): Mixed intercept/evasion; nature depends on initial geometry and \(\alpha, \beta\).
*   **(c)** \(\alpha \gg 1, \beta \approx 1\): No evasion by T1; engagement between D2/A3 depends on geometry/\(\beta\).
*   **(d)** \(\beta \gg 1, \alpha \approx 1\): No evasion by A3; engagement between A3/T1 depends on geometry/\(\alpha\).

### 4.4.4 Game Termination Criteria and Outcomes
**Termination Criteria:** Minimum miss distance (MD).
*   **MD23:** Miss distance between Defender 2 and Attacker 3.
*   **MD31:** Miss distance between Attacker 3 and Target 1.

**Scenario Progression:** MD23 condition reached first (time \(t_{f_2}\)), followed by MD31 (time \(t_{f_1}\)).

**Outcomes:**
1.  **At \(t = t_{f_2}\) (MD23 reached):**
    *   **Defender 2 Wins:** If MD23 < Defender 2's warhead lethal radius \(\implies\) Attacker 3 destroyed. \(\implies\) Target 1 wins (threat eliminated).
    *   **Defender 2 Loses:** If MD23 > Lethal radius \(\implies\) Attacker 3 evades and continues attack.
2.  **At \(t = t_{f_1}\) (MD31 reached - only if Defender 2 lost):**
    *   **Attacker 3 Wins:** If MD31 < Attacker 3's warhead lethal radius \(\implies\) Target 1 destroyed.
    *   **Attacker 3 Loses (Target 1 Wins):** If MD31 > Lethal radius.

## 4.5 Discussion and Conclusions
This chapter developed a **three-party differential game** framework for a missile defense scenario: a high-value target (aircraft) fires a defender missile while evading, and an attacker missile pursues the target while evading the defender.

**Key Developments:**
1.  **3-D Engagement Kinematics Model:** Derived for the three-vehicle scenario.
2.  **LQPI Formulation:** Defined separate PIs for the two engagements (T1-A3, D2-A3), incorporating conflicting objectives via negative evasion input weights.
3.  **Optimal Guidance Laws:** Derived using Hamiltonian and necessary conditions. Solutions involve:
    *   **Matrix Riccati DE (MRDE):** For state feedback gains.
    *   **Vector Riccati DE (VRDE):** For disturbance input effects.
4.  **Analytical Solutions:** Obtained for MRDE and VRDE (under piecewise constant disturbance assumption) as functions of time-to-go (\(T_i\)) and composite parameters (\(\gamma_i = r_{31}, r_{23}\)).
5.  **Guidance Law Structure:** Comprises state feedback and disturbance compensation terms.
6.  **Rule-Based AI:** Proposed for scheduling additional evasion maneuvers (e.g., step, sinusoidal, random accelerations) via disturbance inputs \(a_i^d\), and for PI weighting switching based on time-to-go/miss-distance to deceive adversaries.

**Requirements & Limitations:**
*   \(r^e > r^p\) (Evasion weight > Pursuit weight) required for Riccati solution existence.
*   Guidance commands derived in fixed axis; application in body axis makes them sub-optimal.
*   Autopilot dynamics not included.
*   Full state information assumed.

**Future Work:**
1.  Test guidance for different PI weightings.
2.  Evaluate performance with different vehicle characteristics (velocities, accelerations, autopilot bandwidths).
3.  Study scenarios with limited/inaccurate state or time-to-go information.
4.  Implement and test using a 3-D simulation platform (Chapter 5) to assess performance, sensitivity, and robustness.
5.  Evaluate vulnerability of existing systems and enhance future missiles using game-theoretic "intelligent" guidance in realistic combat environments.

## References
1.  Ben-Asher, J. Z., Isaac, Y., *Advances in Missile Guidance Theory*, Vol. 180, AIAA, 1998.
2.  Isaacs, R., *Differential Games*, Dover Publications, 1965.
3.  Robb, M. C., Tsourdos, A., White, B. A., "Earliest Intercept Line Guidance Using a Game Theory Approach," AIAA GNC Conf., AIAA-2006-6216, 2006.
4.  Sage, A. P., *Optimum Systems Control*, Prentice Hall, 1968.
5.  Shinar, J., Siegel, A., Gold, Y., "On the analysis of a complex differential game using artificial intelligence," IEEE CDC, pp. 1436–1441, 1988.
6.  Shinar, J., Guelman, M., Silberman, G., Green, A., "On optimal missile avoidance — comparison between optimal control and differential game solutions," IEEE CCA, pp. 453–459, 1989.
7.  Shinar, J., Shima, T., "A game theoretical interceptor guidance law for ballistic missile defence," IEEE CDC, pp. 2780–2785, 1996.
8.  Shinar, J., "On the feasibility of 'hit to kill' in the interception of a manoeuvring target," Amer. Ctrl. Conf., pp. 3358–3363, 2001.
9.  Zarchan, P., *Tactical and Strategic Missile Guidance*, 2nd ed., Vol. 199, AIAA, 2002.

# Appendix
*(Derivations follow closely the structure and methods outlined in the original document, primarily involving substitution of assumed solutions into the Riccati DEs and kinematics equations, followed by algebraic manipulation to verify the solutions presented in Sections 4.3 and 4.4.)*

## A4.1 Derivation of the Riccati Equations
*(Substitution of (4.3.15, 4.3.16) into (4.2.11) yields (A4.1.1). Substitution of (4.3.17, 4.3.18) into (4.2.13) yields (A4.1.2). Substitution of \(\lambda\) forms into (4.3.13, 4.3.14) yields (A4.1.3, A4.1.4). Substituting (A4.1.1, A4.1.2) into (A4.1.3, A4.1.4), simplifying, and setting coefficients of \(y\) to zero leads to the MRDE (A4.1.7, A4.1.9) and VRDE (A4.1.8, A4.1.10), equivalent to (4.3.19-4.3.22).)*

## A4.2 Analytical Solution for Riccati Differential Equations
*(Presents the solution matrix [Pi] (A4.2.1) and elements (A4.2.2-A4.2.4) for the specific case, matching (4.4.1-4.4.4). Reiterates existence condition \(r^e > r^p\).)*

## A4.3 State Feedback Gains
*(Defines state feedback gains (A4.3.1-A4.3.4) and provides their explicit matrix forms (A4.3.5-A4.3.8), matching (4.4.2-4.4.7).)*

## A4.4 Disturbance Inputs
*(Derives the solutions for the VRDE (η components) under the piecewise constant disturbance assumption, matching the solutions given in (4.4.8-4.4.19).)*

## A4.5 Guidance Disturbance Inputs
*(Provides the expressions for the disturbance components of the guidance commands (k^e_1, k^p_3, k^p_2, k^e_3), matching (4.4.20-4.4.23).)*