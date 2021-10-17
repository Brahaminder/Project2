# Project2
This project is based on the Multi-Depot Vehicle Routing Problem.The MDVRP is formulated using a vehicle-flow and a set-partitioning formulation, both of which are exploited at different stages of the algorithm. The lower bound computed with the vehicle-flow formulation is used to eliminate non-promising edges, thus reducing the complexity of the pricing sub-problem used to solve the set-partitioning formulation. Several classes of valid inequalities are added to strengthen both formulations, including a new family of valid inequalities used to forbid cycles of an arbitrary length. To validate our approach, we also consider the capacitated vehicle routing problem (CVRP) as a particular case of the MDVRP, and conduct extensive computational experiments on several instances from the literature to show its effectiveness. The computational results show that the proposed algorithm is competitive against state-of-the-art methods for these two classes of vehicle routing problems, and is able to solve to optimality some previously open instances. Moreover, for the instances that cannot be solved by the proposed algorithm, the final lower bounds prove stronger than those obtained by earlier methods.
