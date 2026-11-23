"""
Account Partitioning module for CFP simulation.
Groups frequently-interacting accounts into same shard.

Uses Louvain algorithm for community detection.
Reference: Blondel et al., "Fast unfolding of communities in large networks", 2008

From paper section 4.8: minimize cross-shard transactions
by clustering accounts based on interaction patterns.
"""

import random
from collections import defaultdict
import math


class InteractionGraph:
    """
    Graph of account interactions.
    Edges weighted by transaction frequency.
    """

    def __init__(self):
        self.edges = defaultdict(lambda: defaultdict(float))
        self.accounts = set()
        self.total_weight = 0

    def add_interaction(self, account1, account2, weight=1):
        """Record interaction between two accounts."""
        if account1 == account2:
            return

        self.edges[account1][account2] += weight
        self.edges[account2][account1] += weight
        self.accounts.add(account1)
        self.accounts.add(account2)
        self.total_weight += weight

    def get_interaction_weight(self, account1, account2):
        """Get interaction weight between two accounts."""
        return self.edges[account1].get(account2, 0)

    def get_neighbors(self, account):
        """Get all accounts that interacted with this one."""
        return list(self.edges[account].keys())

    def get_degree(self, account):
        """Get sum of all edge weights for an account."""
        return sum(self.edges[account].values())

    def get_edge_weight_sum(self):
        """Get total weight of all edges (m in Louvain)."""
        return self.total_weight


class LouvainClustering:
    """
    Louvain algorithm for community detection.

    Optimizes modularity: Q = (1/2m) * sum[(A_ij - k_i*k_j/2m) * delta(c_i, c_j)]

    where:
    - m = total edge weight
    - A_ij = edge weight between i and j
    - k_i = degree of node i
    - c_i = community of node i
    - delta = 1 if same community, 0 otherwise
    """

    def __init__(self, graph, target_clusters=None):
        """
        Initialize Louvain clustering.

        graph: InteractionGraph object
        target_clusters: desired number of clusters (shards)
        """
        self.graph = graph
        self.target_clusters = target_clusters
        self.communities = {}  # node -> community_id
        self.community_members = defaultdict(set)  # community_id -> set of nodes

    def initialize_communities(self):
        """Start with each node in its own community."""
        for i, node in enumerate(self.graph.accounts):
            self.communities[node] = i
            self.community_members[i].add(node)

    def get_community_degree(self, community_id):
        """Get sum of degrees for all nodes in community."""
        total = 0
        for node in self.community_members[community_id]:
            total += self.graph.get_degree(node)
        return total

    def get_edges_to_community(self, node, community_id):
        """Get sum of edge weights from node to community."""
        total = 0
        for member in self.community_members[community_id]:
            total += self.graph.get_interaction_weight(node, member)
        return total

    def calculate_modularity_gain(self, node, new_community):
        """
        Calculate modularity gain from moving node to new_community.

        Delta Q = [sum_in + k_i_in]/2m - [(sum_tot + k_i)/2m]^2
                - [sum_in/2m - (sum_tot/2m)^2 - (k_i/2m)^2]

        Simplified:
        Delta Q = [k_i_in/m] - [k_i * sum_tot / (2m^2)]
        """
        m = self.graph.get_edge_weight_sum()
        if m == 0:
            return 0

        k_i = self.graph.get_degree(node)
        k_i_in = self.get_edges_to_community(node, new_community)
        sum_tot = self.get_community_degree(new_community)

        # if node is already in this community, subtract its contribution
        if self.communities.get(node) == new_community:
            sum_tot -= k_i

        delta_q = (k_i_in / m) - (k_i * sum_tot / (2 * m * m))
        return delta_q

    def move_node(self, node, new_community):
        """Move node to a new community."""
        old_community = self.communities.get(node)

        if old_community is not None:
            self.community_members[old_community].discard(node)
            # clean up empty communities
            if len(self.community_members[old_community]) == 0:
                del self.community_members[old_community]

        self.communities[node] = new_community
        self.community_members[new_community].add(node)

    def phase_one(self):
        """
        First phase: local moving of nodes.
        Move each node to community that maximizes modularity gain.
        """
        improved = True
        iterations = 0
        max_iterations = 100

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            nodes = list(self.graph.accounts)
            random.shuffle(nodes)

            for node in nodes:
                current_community = self.communities[node]
                best_community = current_community
                best_gain = 0

                # try moving to neighbor's communities
                neighbor_communities = set()
                for neighbor in self.graph.get_neighbors(node):
                    neighbor_communities.add(self.communities[neighbor])

                for community in neighbor_communities:
                    if community == current_community:
                        continue

                    gain = self.calculate_modularity_gain(node, community)
                    if gain > best_gain:
                        best_gain = gain
                        best_community = community

                if best_community != current_community and best_gain > 0:
                    self.move_node(node, best_community)
                    improved = True

        return iterations

    def merge_small_communities(self):
        """
        Merge communities to reach target number of clusters.
        Merges smallest communities with their most connected neighbor.
        """
        if self.target_clusters is None:
            return

        while len(self.community_members) > self.target_clusters:
            # find smallest community
            smallest = min(self.community_members.items(),
                          key=lambda x: len(x[1]))
            smallest_id = smallest[0]
            smallest_members = smallest[1]

            if len(smallest_members) == 0:
                del self.community_members[smallest_id]
                continue

            # find most connected other community
            community_connections = defaultdict(float)

            for member in smallest_members:
                for neighbor in self.graph.get_neighbors(member):
                    neighbor_comm = self.communities[neighbor]
                    if neighbor_comm != smallest_id:
                        weight = self.graph.get_interaction_weight(member, neighbor)
                        community_connections[neighbor_comm] += weight

            if not community_connections:
                # no connections, merge with largest
                other_comms = [c for c in self.community_members.keys()
                             if c != smallest_id]
                if not other_comms:
                    break
                target = max(other_comms,
                           key=lambda c: len(self.community_members[c]))
            else:
                target = max(community_connections.items(),
                           key=lambda x: x[1])[0]

            # merge
            for member in list(smallest_members):
                self.move_node(member, target)

    def reindex_communities(self):
        """Reindex communities to be 0, 1, 2, ..., n-1"""
        old_to_new = {}
        new_id = 0

        for old_id in sorted(self.community_members.keys()):
            if len(self.community_members[old_id]) > 0:
                old_to_new[old_id] = new_id
                new_id += 1

        # update communities
        new_communities = {}
        new_members = defaultdict(set)

        for node, old_comm in self.communities.items():
            new_comm = old_to_new.get(old_comm, 0)
            new_communities[node] = new_comm
            new_members[new_comm].add(node)

        self.communities = new_communities
        self.community_members = new_members

    def fit(self):
        """
        Run full Louvain algorithm.
        Returns dict mapping node -> cluster_id
        """
        self.initialize_communities()

        # phase 1: optimize modularity
        iterations = self.phase_one()

        # merge to target cluster count if specified
        if self.target_clusters:
            self.merge_small_communities()

        # clean up indices
        self.reindex_communities()

        return self.communities

    def get_modularity(self):
        """
        Calculate current modularity score.
        Q = (1/2m) * sum[(A_ij - k_i*k_j/2m) * delta(c_i, c_j)]
        """
        m = self.graph.get_edge_weight_sum()
        if m == 0:
            return 0

        q = 0
        for node_i in self.graph.accounts:
            k_i = self.graph.get_degree(node_i)
            c_i = self.communities.get(node_i)

            for node_j in self.graph.accounts:
                if self.communities.get(node_j) != c_i:
                    continue

                k_j = self.graph.get_degree(node_j)
                a_ij = self.graph.get_interaction_weight(node_i, node_j)

                q += a_ij - (k_i * k_j / (2 * m))

        return q / (2 * m)


class AccountPartitioner:
    """
    Main partitioner that reorganizes accounts across shards.
    Uses Louvain algorithm for optimal clustering.
    """

    def __init__(self, network):
        self.network = network
        self.graph = InteractionGraph()
        self.migration_history = []
        self.modularity_scores = []

    def record_transaction(self, sender, receiver):
        """Record a transaction for analysis."""
        self.graph.add_interaction(sender, receiver)

    def record_transactions_batch(self, transactions):
        """Record multiple transactions."""
        for tx in transactions:
            self.record_transaction(tx.sender, tx.receiver)

    def analyze_current_distribution(self):
        """Analyze how well accounts are distributed."""
        cross_shard = 0
        same_shard = 0

        for acc1 in self.graph.accounts:
            shard1 = self.network.get_account_shard(acc1)
            if shard1 is None:
                continue

            for acc2, weight in self.graph.edges[acc1].items():
                shard2 = self.network.get_account_shard(acc2)
                if shard2 is None:
                    continue

                if shard1 == shard2:
                    same_shard += weight
                else:
                    cross_shard += weight

        # each edge counted twice
        cross_shard = cross_shard / 2
        same_shard = same_shard / 2

        total = cross_shard + same_shard
        cross_ratio = cross_shard / total if total > 0 else 0

        return {
            "cross_shard": int(cross_shard),
            "same_shard": int(same_shard),
            "total": int(total),
            "cross_ratio": cross_ratio
        }

    def repartition(self):
        """
        Repartition accounts using Louvain algorithm.
        Returns number of migrations.
        """
        # run Louvain
        louvain = LouvainClustering(self.graph, target_clusters=self.network.num_shards)
        clusters = louvain.fit()

        # record modularity
        modularity = louvain.get_modularity()
        self.modularity_scores.append(modularity)

        # migrate accounts
        migrations = 0

        for account, new_shard in clusters.items():
            current_shard = self.network.get_account_shard(account)

            if current_shard is not None and current_shard != new_shard:
                self.migrate_account(account, current_shard, new_shard)
                migrations += 1

        self.migration_history.append({
            "migrations": migrations,
            "accounts": len(clusters),
            "modularity": modularity
        })

        return migrations

    def migrate_account(self, account, from_shard, to_shard):
        """Move account from one shard to another."""
        if from_shard in self.network.shards:
            self.network.shards[from_shard].accounts.discard(account)

        if to_shard in self.network.shards:
            self.network.shards[to_shard].accounts.add(account)

        self.network.account_to_shard[account] = to_shard

    def get_improvement(self, before_stats, after_stats):
        """Calculate improvement in cross-shard ratio."""
        before_ratio = before_stats["cross_ratio"]
        after_ratio = after_stats["cross_ratio"]

        if before_ratio == 0:
            return 0

        improvement = (before_ratio - after_ratio) / before_ratio
        return improvement

    def get_modularity(self):
        """Get current modularity score."""
        louvain = LouvainClustering(self.graph, target_clusters=self.network.num_shards)

        # set communities based on current shard assignment
        for account in self.graph.accounts:
            shard = self.network.get_account_shard(account)
            if shard is not None:
                louvain.communities[account] = shard
                louvain.community_members[shard].add(account)

        return louvain.get_modularity()

    def print_analysis(self):
        """Print current distribution analysis."""
        stats = self.analyze_current_distribution()

        print(f"\nAccount Distribution Analysis:")
        print(f"  Total interactions: {stats['total']}")
        print(f"  Same-shard: {stats['same_shard']} ({1-stats['cross_ratio']:.1%})")
        print(f"  Cross-shard: {stats['cross_shard']} ({stats['cross_ratio']:.1%})")

        if self.modularity_scores:
            print(f"  Modularity: {self.modularity_scores[-1]:.4f}")


# test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')

    from sharding import Network

    print("Testing Louvain-based Account Partitioning")
    print("=" * 55)

    # setup
    net = Network(num_shards=4, validators_per_shard=25, total_validators=200)

    accounts = [f"user_{i}" for i in range(100)]
    for acc in accounts:
        net.register_account(acc)

    partitioner = AccountPartitioner(net)

    # create clear clusters in transactions
    print("\nCreating clustered transaction pattern...")
    print("  Group 0: user_0 to user_24")
    print("  Group 1: user_25 to user_49")
    print("  Group 2: user_50 to user_74")
    print("  Group 3: user_75 to user_99")

    for _ in range(1000):
        group = random.randint(0, 3)
        base = group * 25

        if random.random() < 0.9:
            s = f"user_{base + random.randint(0, 24)}"
            r = f"user_{base + random.randint(0, 24)}"
        else:
            s = random.choice(accounts)
            r = random.choice(accounts)

        if s != r:
            partitioner.record_transaction(s, r)

    print(f"\nRecorded {partitioner.graph.total_weight} interactions")

    # before
    print("\nBefore Louvain partitioning:")
    before = partitioner.analyze_current_distribution()
    partitioner.print_analysis()

    # repartition
    migrations = partitioner.repartition()
    print(f"\nMigrated {migrations} accounts")

    # after
    print("\nAfter Louvain partitioning:")
    after = partitioner.analyze_current_distribution()
    partitioner.print_analysis()

    # improvement
    improvement = partitioner.get_improvement(before, after)
    print(f"\nImprovement: {improvement:.1%} reduction in cross-shard")

    # check cluster quality
    print("\nCluster distribution:")
    for group in range(4):
        base = group * 25
        shards = {}
        for i in range(25):
            shard = net.get_account_shard(f"user_{base + i}")
            shards[shard] = shards.get(shard, 0) + 1

        main_shard = max(shards.items(), key=lambda x: x[1])
        print(f"  Group {group}: {main_shard[1]}/25 in shard {main_shard[0]}")

    print("\nLouvain partitioning test complete!")

def _account_load(graph):
    """l_A: number of recorded transaction endpoints of account A (edge-weight degree)."""
    return {a: sum(graph.edges[a].values()) for a in graph.accounts}


def repartition_balanced(partitioner, eps=0.10):
    """
    Alg. 1: Louvain communities, then greedy placement in decreasing workload subject to the
    per-shard budget U = (1 + eps) * total_load / K (Eq. 9). A community that fits nowhere is split
    into single accounts. Each unit goes to the eligible shard with the smallest cut weight to
    already-placed neighbours (ties: lower relative load, then lower shard id).
    Returns (migrations, max_load_over_budget).
    """
    net, g = partitioner.network, partitioner.graph
    K = net.num_shards
    louvain = LouvainClustering(g, target_clusters=None)
    clusters = louvain.fit()
    partitioner.modularity_scores.append(louvain.get_modularity())
    load = _account_load(g)
    U = (1 + eps) * sum(load.values()) / K
    comms = {}
    for a, c in clusters.items():
        comms.setdefault(c, []).append(a)
    units = sorted(comms.values(), key=lambda m: (-sum(load[a] for a in m), min(m)))
    placed, L = {}, [0.0] * K
    queue = list(units)
    while queue:
        unit = queue.pop(0)
        lu = sum(load[a] for a in unit)
        elig = [j for j in range(K) if L[j] + lu <= U]
        if not elig:
            if len(unit) == 1:
                elig = [min(range(K), key=lambda j: L[j])]
            else:
                queue = sorted(queue + [[a] for a in unit], key=lambda m: (-sum(load[a] for a in m), min(m)))
                continue
        cost = [0.0] * K
        for a in unit:
            for b, w in g.edges[a].items():
                if b in placed:
                    for j in range(K):
                        if j != placed[b]:
                            cost[j] += w
        j = min(elig, key=lambda j: (cost[j], (L[j] + lu) / U, j))
        for a in unit:
            placed[a] = j
        L[j] += lu
    migrations = 0
    for a, j in placed.items():
        cur = net.get_account_shard(a)
        if cur is not None and cur != j:
            partitioner.migrate_account(a, cur, j)
            migrations += 1
    return migrations, max(L) / U


def shard_load_ratio(partitioner, eps=0.10):
    """Max shard load divided by the Eq. (9) budget for the current placement."""
    net, g = partitioner.network, partitioner.graph
    load = _account_load(g)
    L = [0.0] * net.num_shards
    for a, l in load.items():
        s = net.get_account_shard(a)
        if s is not None:
            L[s] += l
    return max(L) / ((1 + eps) * sum(load.values()) / net.num_shards)
