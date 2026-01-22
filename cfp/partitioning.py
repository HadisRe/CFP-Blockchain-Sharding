"""
Account Partitioning module for CFP simulation.
Groups frequently-interacting accounts into same shard.

Uses Louvain algorithm for community detection.
"""

import random
from collections import defaultdict
import math


class InteractionGraph:
     

    def __init__(self):
        self.edges = defaultdict(lambda: defaultdict(float))
        self.accounts = set()
        self.total_weight = 0

    def add_interaction(self, account1, account2, weight=1):
         if account1 == account2:
            return

        self.edges[account1][account2] += weight
        self.edges[account2][account1] += weight
        self.accounts.add(account1)
        self.accounts.add(account2)
        self.total_weight += weight

    def get_interaction_weight(self, account1, account2):
         return self.edges[account1].get(account2, 0)

    def get_neighbors(self, account):
         return list(self.edges[account].keys())

    def get_degree(self, account):
         return sum(self.edges[account].values())

    def get_edge_weight_sum(self):
         return self.total_weight


class LouvainClustering:
   

    def __init__(self, graph, target_clusters=None):
         
        self.graph = graph
        self.target_clusters = target_clusters
        self.communities = {}  # node -> community_id
        self.community_members = defaultdict(set)  # community_id -> set of nodes

    def initialize_communities(self):
         for i, node in enumerate(self.graph.accounts):
            self.communities[node] = i
            self.community_members[i].add(node)

    def get_community_degree(self, community_id):
         total = 0
        for node in self.community_members[community_id]:
            total += self.graph.get_degree(node)
        return total

    def get_edges_to_community(self, node, community_id):
         total = 0
        for member in self.community_members[community_id]:
            total += self.graph.get_interaction_weight(node, member)
        return total

    def calculate_modularity_gain(self, node, new_community):
        
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
         old_community = self.communities.get(node)

        if old_community is not None:
            self.community_members[old_community].discard(node)
            # clean up empty communities
            if len(self.community_members[old_community]) == 0:
                del self.community_members[old_community]

        self.communities[node] = new_community
        self.community_members[new_community].add(node)

    def phase_one(self):
       
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
        "
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
         if from_shard in self.network.shards:
            self.network.shards[from_shard].accounts.discard(account)

        if to_shard in self.network.shards:
            self.network.shards[to_shard].accounts.add(account)

        self.network.account_to_shard[account] = to_shard

    def get_improvement(self, before_stats, after_stats):
         before_ratio = before_stats["cross_ratio"]
        after_ratio = after_stats["cross_ratio"]

        if before_ratio == 0:
            return 0

        improvement = (before_ratio - after_ratio) / before_ratio
        return improvement

    def get_modularity(self):
         louvain = LouvainClustering(self.graph, target_clusters=self.network.num_shards)

        # set communities based on current shard assignment
        for account in self.graph.accounts:
            shard = self.network.get_account_shard(account)
            if shard is not None:
                louvain.communities[account] = shard
                louvain.community_members[shard].add(account)

        return louvain.get_modularity()

    def print_analysis(self):
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
