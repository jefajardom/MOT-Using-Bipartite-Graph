from torch import nn

class BipartiteNeuralMessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout=0.):
        super().__init__()

        edge_in_dim  = 2*node_dim + 2*edge_dim # 2*edge_dim since we always concatenate initial edge features
        self.edge_mlp = nn.Sequential(*[nn.Linear(edge_in_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout)])

        node_in_dim  = node_dim + edge_dim
        self.node_mlp = nn.Sequential(*[nn.Linear(node_in_dim, node_dim), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(node_dim, node_dim), nn.ReLU(), nn.Dropout(dropout)])

    def edge_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Node-to-edge updates, as descibed in slide 71, lecture 5.
        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, 2 x edge_dim)
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_a_embeds: torch.Tensor with shape (|B|, node_dim)

        returns:
            updated_edge_feats = torch.Tensor with shape (|A|, |B|, edge_dim)
        """

        n_nodes_a, n_nodes_b, _  = edge_embeds.shape

        ########################
        #### TODO starts
        ########################

        na = nodes_a_embeds.view(nodes_a_embeds.shape[0], 1, nodes_a_embeds.shape[1]).repeat(1, nodes_b_embeds.shape[0], 1)
        nb = nodes_b_embeds.view(1, nodes_b_embeds.shape[0], nodes_b_embeds.shape[1]).repeat(nodes_a_embeds.shape[0], 1, 1)
        edge_in = torch.cat((edge_embeds, na, nb), dim=2) # has shape (|A|, |B|, 2*node_dim + 2*edge_dim)

        ########################
        #### TODO ends
        ########################


        return self.edge_mlp(edge_in)

    def node_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Edge-to-node updates, as descibed in slide 75, lecture 5.

        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, edge_dim)
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)

        returns:
            tuple(
                updated_nodes_a_embeds: torch.Tensor with shape (|A|, node_dim),
                updated_nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)
                )
        """

        ########################
        #### TODO starts
        ########################

        # NOTE: Use 'sum' as aggregation function


        nodes_a_in = torch.cat((nodes_a_embeds, torch.sum(edge_embeds, dim=1)), dim=1) # Has shape (|A|, node_dim + edge_dim)
        nodes_b_in = torch.cat((nodes_b_embeds, torch.sum(edge_embeds, dim=0)), dim=1) # Has shape (|B|, node_dim + edge_dim)

        ########################
        #### TODO ends
        ########################

        nodes_a = self.node_mlp(nodes_a_in)
        nodes_b = self.node_mlp(nodes_b_in)

        return nodes_a, nodes_b

    def forward(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        edge_embeds_latent = self.edge_update(edge_embeds, nodes_a_embeds, nodes_b_embeds)
        nodes_a_latent, nodes_b_latent = self.node_update(edge_embeds_latent, nodes_a_embeds, nodes_b_embeds)

        return edge_embeds_latent, nodes_a_latent, nodes_b_latent



#################

from torch.nn import functional as F
class AssignmentSimilarityNet(nn.Module):
    def __init__(self, reid_network, node_dim, edge_dim, reid_dim, edges_in_dim, num_steps, dropout=0.4):
        super().__init__()
        self.reid_network = reid_network
        self.graph_net = BipartiteNeuralMessagePassingLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout)
        self.num_steps = num_steps
        self.cnn_linear = nn.Linear(reid_dim, node_dim)
        self.edge_in_mlp = nn.Sequential(*[nn.Linear(edges_in_dim, edge_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(edge_dim, edge_dim), nn.ReLU(),nn.Dropout(dropout)])
        self.classifier = nn.Sequential(*[nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, 1)])


    def compute_edge_feats(self, track_coords, current_coords, track_t, curr_t):
        """
        Computes initial edge feature tensor

        Args:
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)

            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)


        Returns:
            tensor with shape (num_trakcs, num_boxes, 5) containing pairwise
            position and time difference features
        """

        ########################
        #### TODO starts
        ########################

        # NOTE 1: we recommend you to use box centers to compute distances
        # in the x and y coordinates.

        # NOTE 2: Check out the  code inside train_one_epoch function and
        # LongTrackTrainingDataset class a few cells below to debug this
        ###
        track_center_coords = track_coords.view(track_coords.shape[0], 2, 2).mean(1)
        current_center_coords = current_coords.view(current_coords.shape[0], 2, 2).mean(1)
        edge_feats = torch.zeros(track_coords.shape[0], current_coords.shape[0], 5)
        for i in range(track_coords.shape[0]):
          for j in range(current_coords.shape[0]):
            wi = track_coords[i, 2] - track_coords[i, 0]
            wj = current_coords[j, 2] - current_coords[j, 0]
            hi = track_coords[i, 3] - track_coords[i, 1]
            hj = current_coords[j, 3] - current_coords[j, 1]
            xi = track_center_coords[i, 0]
            yi = track_center_coords[i, 1]
            xj = current_center_coords[j, 0]
            yj = current_center_coords[j, 1]
            edge_feats[i, j, 4] = curr_t[j] - track_t[i]
            edge_feats[i, j, 2] = torch.log(hi / hj)
            edge_feats[i, j, 3] = torch.log(wi / wj)
            edge_feats[i, j, 0] = 2 * (xj - xi) / (hj + hi)
            edge_feats[i, j, 1] = 2 * (yj - yi) / (hj + hi)

        edge_feats = edge_feats.to(track_coords.device)


        ########################
        #### TODO ends
        ########################

        return edge_feats # has shape (num_trakcs, num_boxes, 5)


    def forward(self, track_app, current_app, track_coords, current_coords, track_t, curr_t):
        """
        Args:
            track_app: track's reid embeddings, torch.Tensor with shape (num_tracks, 512)
            current_app: current frame detections' reid embeddings, torch.Tensor with shape (num_boxes, 512)
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)

            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)

        Returns:
            classified edges: torch.Tensor with shape (num_steps, num_tracks, num_boxes),
                             containing at entry (step, i, j) the unnormalized probability that track i and
                             detection j are a match, according to the classifier at the given neural message passing step
        """

        # Get initial edge embeddings to
        dist_reid = cosine_distance(track_app, current_app)
        pos_edge_feats = self.compute_edge_feats(track_coords, current_coords, track_t, curr_t)
        edge_feats = torch.cat((pos_edge_feats, dist_reid.unsqueeze(-1)), dim=-1)
        edge_embeds = self.edge_in_mlp(edge_feats)
        initial_edge_embeds = edge_embeds.clone()

        # Get initial node embeddings, reduce dimensionality from 512 to node_dim
        track_embeds = F.relu(self.cnn_linear(track_app))
        curr_embeds =F.relu(self.cnn_linear(current_app))

        classified_edges = []
        for _ in range(self.num_steps):
            edge_embeds = torch.cat((edge_embeds, initial_edge_embeds), dim=-1)
            edge_embeds, track_embeds, curr_embeds = self.graph_net(edge_embeds=edge_embeds,
                                                                    nodes_a_embeds=track_embeds,
                                                                    nodes_b_embeds=curr_embeds)

            classified_edges.append(self.classifier(edge_embeds))

        return torch.stack(classified_edges).squeeze(-1)



class NeuralMessagePassingTracker(LongVersion_HungarianTracker):
    def __init__(self, assign_net, *args, **kwargs):
        self.assign_net = assign_net
        super().__init__(*args, **kwargs)

    def data_association(self, boxes, scores, pred_features):
        if self.tracks:
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0).cuda()
            track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0).cuda()

            # Hacky way to recover the timestamps of boxes and tracks
            curr_t = self.im_index * torch.ones((pred_features.shape[0],)).cuda()
            track_t = torch.as_tensor([self.im_index - t.inactive - 1 for t in self.tracks]).cuda()

            ########################
            #### TODO starts
            ########################

            # Do a forward pass through self.assign_net to obtain our costs.
            # Note: self.assign_net will return unnormalized probabilities.
            # Make sure to apply the sigmoid function to them!


            fwd = self.assign_net.forward(track_features, pred_features.cuda(), track_boxes, boxes.cuda(), track_t, curr_t)
            pred_sim = torch.sigmoid(fwd).cpu().data.numpy()


            ########################
            #### TODO ends
            ########################

            pred_sim = pred_sim[-1]  # Use predictions at last message passing step
            distance = (1- pred_sim)

            # Do not allow mataches when sim < 0.5, to avoid low-confident associations
            distance = np.where(pred_sim < 0.5, _UNMATCHED_COST, distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx,distance, boxes, scores, pred_features)


        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)
