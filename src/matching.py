import collections
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import motmetrics as mm
mm.lap.default_solver = 'lap'

import market.metrics as metrics

import os.path as osp


_UNMATCHED_COST=355


#Function Build cost matrix.
def ltrb_to_ltwh(ltrb_boxes):
    ltwh_boxes = copy.deepcopy(ltrb_boxes)
    ltwh_boxes[:, 2] = ltrb_boxes[:, 2] - ltrb_boxes[:, 0]
    ltwh_boxes[:, 3] = ltrb_boxes[:, 3] - ltrb_boxes[:, 1]
    return ltwh_boxes


class Tracker:
    """The main tracking file, here is where magic happens."""
    def __init__(self, obj_detect):
        self.obj_detect = obj_detect

        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

        self.mot_accum = None

    def reset(self, hard=True):
        self.tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def add(self, new_boxes, new_scores):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(Track(
                new_boxes[i],
                new_scores[i],
                self.track_num + i
            ))
        self.track_num += num_new

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            box = self.tracks[0].box
        elif len(self.tracks) > 1:
            box = torch.stack([t.box for t in self.tracks], 0)
        else:
            box = torch.zeros(0).cuda()
        return box

    def update_results(self):
        # results
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1


    def data_association(self, boxes, scores):
        self.tracks = []
        self.add(boxes, scores)

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        #boxes, scores = self.obj_detect.detect(frame['img'])
        boxes, scores = frame['det']['boxes'], frame['det']['scores']
        self.data_association(boxes, scores)
        self.update_results()
        
    def get_results(self):
        return self.results


class ReID_Tracker(Tracker):
    def add(self, new_boxes, new_scores, new_features):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(Track(
                new_boxes[i],
                new_scores[i],
                self.track_num + i,
                new_features[i]
            ))
        self.track_num += num_new

    def reset(self, hard=True):
        self.tracks = []
        #self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def data_association(self, boxes, scores, features):
        raise NotImplementedError

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        boxes = frame['det']['boxes']
        scores = frame['det']['scores']
        reid_feats= frame['det']['reid'].cpu()
        self.data_association(boxes, scores, reid_feats)

        # results
        self.update_results()


    def compute_distance_matrix(self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0):
        UNMATCHED_COST = 255.0

        # Build cost matrix.
        distance = mm.distances.iou_matrix(ltrb_to_ltwh(track_boxes).numpy(), ltrb_to_ltwh(boxes).numpy(), max_iou=0.5)

        appearance_distance = metrics.compute_distance_matrix(track_features, pred_features, metric_fn=metric_fn)
        appearance_distance = appearance_distance.numpy() * 0.5
        # return appearance_distance

        assert np.alltrue(appearance_distance >= -0.1)
        assert np.alltrue(appearance_distance <= 1.1)

        combined_costs = alpha * distance + (1-alpha) * appearance_distance

        # Set all unmatched costs to _UNMATCHED_COST.
        distance = np.where(np.isnan(distance), UNMATCHED_COST, combined_costs)

        distance = np.where(appearance_distance > 0.1, UNMATCHED_COST, distance)

        return distance


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, box, score, track_id, feature=None, inactive=0):
        self.id = track_id
        self.box = box
        self.score = score
        self.feature = collections.deque([feature])
        self.inactive = inactive
        self.max_features_num = 10


    def add_feature(self, feature):
        """Adds new appearance features to the object."""
        self.feature.append(feature)
        if len(self.feature) > self.max_features_num:
            self.feature.popleft()

    def get_feature(self):
        if len(self.feature) > 1:
            feature = torch.stack(list(self.feature), dim=0)
        else:
            feature = self.feature[0].unsqueeze(0)
        #return feature.mean(0, keepdim=False)
        return feature[-1]


class Hungarian_Tracker(ReID_Tracker):
    def data_association(self, boxes, scores, pred_features):
        """Refactored from previous implementation to split it onto distance computation and track management"""
        if self.tracks:
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0)
            track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)

            distance = self.compute_distance_matrix(track_features, pred_features,
                                                    track_boxes, boxes, metric_fn=cosine_distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx,distance, boxes, scores, pred_features)


        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)

    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
        """Updates existing tracks and removes unmatched tracks.
           Reminder: If the costs are equal to _UNMATCHED_COST, it's not a
           match.
        """
        track_ids = [t.id for t in self.tracks]

        unmatched_track_ids = []
        seen_track_ids = []
        seen_box_idx = []
        for track_idx, box_idx in zip(row_idx, col_idx):
            costs = distance[track_idx, box_idx]
            internal_track_id = track_ids[track_idx]
            seen_track_ids.append(internal_track_id)
            if costs == _UNMATCHED_COST:
                unmatched_track_ids.append(internal_track_id)
            else:
                self.tracks[track_idx].box = boxes[box_idx]
                self.tracks[track_idx].add_feature(pred_features[box_idx])
                seen_box_idx.append(box_idx)

        unseen_track_ids = set(track_ids) - set(seen_track_ids)
        unmatched_track_ids.extend(list(unseen_track_ids))
        self.tracks = [t for t in self.tracks
                       if t.id not in unmatched_track_ids]


        # Add new tracks.
        new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)


class LongVersion_HungarianTracker(Hungarian_Tracker):
    def __init__(self, patience, *args, **kwargs):
        """ Add a patience parameter"""
        self.patience=patience
        super().__init__(*args, **kwargs)

    def update_results(self):
        """Only store boxes for tracks that are active"""
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            if t.inactive == 0: # Only change
                self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1

    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
        track_ids = [t.id for t in self.tracks]

        unmatched_track_ids = []
        seen_track_ids = []
        seen_box_idx = []
        for track_idx, box_idx in zip(row_idx, col_idx):
            costs = distance[track_idx, box_idx]
            internal_track_id = track_ids[track_idx]
            seen_track_ids.append(internal_track_id)
            if costs == _UNMATCHED_COST:
                unmatched_track_ids.append(internal_track_id)

            else:
                self.tracks[track_idx].box = boxes[box_idx]
                self.tracks[track_idx].add_feature(pred_features[box_idx])

                # Note: the track is matched, therefore, inactive is set to 0
                self.tracks[track_idx].inactive=0
                seen_box_idx.append(box_idx)


        unseen_track_ids = set(track_ids) - set(seen_track_ids)
        unmatched_track_ids.extend(list(unseen_track_ids))
        ##################
        ### TODO starts
        ##################

        # Update the `inactive` attribute for those tracks that have been
        # not been matched. kill those for which the inactive parameter
        # is > self.patience

        records = []
        #iter over self.tracks
        for track in self.tracks:
          if track.id in unmatched_track_ids:
            track.inactive = track.inactive + 1
          if track.inactive > self.patience:
            records.append(track.id)

        self.tracks = [x for x in self.tracks if x.id not in records] # <-- Needs to be updated

        ##################
        ### TODO ends
        ##################

        new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)




