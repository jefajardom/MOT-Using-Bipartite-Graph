from gnn.dataset import LongTrackTrainingDataset
from torch.utils.data import DataLoader
from gnn.trainer import train_one_epoch

MAX_PATIENCE = 30
MAX_EPOCHS = 5
EVAL_FREQ = 1


# Define our model, and init
assign_net = AssignmentSimilarityNet(reid_network=None, # Not needed since we work with precomputed features
                                     node_dim=32,
                                     edge_dim=64,
                                     reid_dim=512,
                                     edges_in_dim=6,
                                     num_steps=10).cuda()

# We only keep two sequences for validation. You can
dataset = LongTrackTrainingDataset(dataset='MOT16-train_wo_val2',
                                   db=train_db,
                                   root_dir= osp.join(root_dir, 'data/MOT16'),
                                   max_past_frames = MAX_PATIENCE,
                                   vis_threshold=0.25)

data_loader = DataLoader(dataset, batch_size=8, collate_fn = lambda x: x,
                         shuffle=True, num_workers=2, drop_last=True)
device = torch.device('cuda')
optimizer = torch.optim.Adam(assign_net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)


best_idf1 = 0.
for epoch in range(1, MAX_EPOCHS + 1):
    print(f"-------- EPOCH {epoch:2d} --------")
    train_one_epoch(model = assign_net, data_loader=data_loader, optimizer=optimizer, print_freq=100)
    scheduler.step()

    if epoch % EVAL_FREQ == 0:
        tracker =  NeuralMessagePassingTracker(assign_net=assign_net.eval(), obj_detect=None, patience=MAX_PATIENCE)
        val_sequences = MOT16Sequences('MOT16-val2', osp.join(root_dir, 'data/MOT16'), vis_threshold=0.)
        res = run_tracker(val_sequences, db=train_db, tracker=tracker, output_dir=None)
        idf1 = res.loc['OVERALL']['idf1']
        if idf1 > best_idf1:
            best_idf1 = idf1
            torch.save(assign_net.state_dict(), osp.join(root_dir, 'output', 'best_ckpt.pth'))