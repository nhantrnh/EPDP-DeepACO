# EPDP-DeepACO

**Authors:**
*   Thanh-Nhan Truong
*   Thuy-Anh Ma
*   Duy-Hoang Tran

## Overview

This repository presents **DeepACO**, a novel framework designed to tackle the **Electric Pickup and Delivery Problem (EPDP)**. Our approach enhances the traditional **Ant Colony Optimization (ACO)** algorithm with **Deep Reinforcement Learning (DRL)** to construct highly efficient and feasible routes.

The core objective is to replace manually designed, problem-specific heuristics-which require significant domain expertise and tuning-with a neural network that automatically learns effective guidance policies.

## Key Contributions and Innovations

Our method introduces the following key improvements and novelties:

### 1. Application of DeepACO to the Complex EPDP
*   We extend the **DeepACO** framework to solve the EPDP, a challenging problem variant.
*   The model is designed to simultaneously handle multiple complex constraints inherent to the EPDP:
    *   **Precedence:** Pickup nodes must be visited before their corresponding delivery nodes.
    *   **Finite Battery Capacity:** The vehicle's battery level must not drop below zero.
    *   **Strategic Recharging:** The model must intelligently decide when and where to recharge to maintain route feasibility.

### 2. Automatic Heuristic Learning via Deep Reinforcement Learning
*   **Core Novelty:** Instead of relying on traditional hand-crafted heuristics (e.g., inverse distance), we employ a **Graph Attention Network (GATv2)** to learn the "desirability" of edges directly from the problem structure.
*   **Benefits:**
    *   **Eliminates Manual Design:** Removes the need for expert knowledge and time-consuming tuning of heuristic functions.
    *   **Highly Adaptive:** The learned heuristic captures complex, non-linear relationships, leading to more powerful and context-aware guidance for the ants.

### 3. Neural-Enhanced Large Neighborhood Search (LNS)
*   We integrate a powerful local search method, **Large Neighborhood Search (LNS)**, which is guided by the learned neural network.
*   During the "recreate" phase of LNS, the algorithm uses a **hybrid cost matrix**. This matrix is a weighted combination of the real travel distance and the heuristic values predicted by our neural model.
*   **Result:** This neural guidance helps the search escape local optima and discover globally superior solutions that a purely greedy approach might miss.

### 4. Superior Performance and Scalability
*   Experimental results demonstrate that **DeepACO** significantly outperforms traditional ACO algorithms (ACS, EAS, MMAS) in both solution quality (lower cost) and computational efficiency (faster runtime).
*   The performance gap widens on larger-scale instances, showcasing our framework's excellent scalability.
*   When benchmarked against a strong solver like **Google OR-Tools** under time limits, DeepACO proves highly competitive and often finds better solutions, especially for complex, large-scale scenarios.
