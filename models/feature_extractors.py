import fairseq

class Wave2vec2FeatureExtractor(pl.LightningModule):
    def __init__(self, model_path):
        super(Wave2vec2FeatureExtractor, self).__init__()

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(model_path)

        self.emb_func = model[0]
        
        if conf["feature"]["freeze"]:
            for param in self.emb_func.parameters():
                param.requires_grad = False

        self.aggregator = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(conf["feature"]["embed_dim"], conf["classifier"]["num_out_classes"], bias=True)
        
#         self.fc = nn.Sequential(
#             nn.Linear(conf["feature"]["embed_dim"], conf["classifier"]["ff_hid"], bias=True),
#             nn.BatchNorm1d(conf["classifier"]["ff_hid"]),
#             nn.ReLU(),
#             nn.Linear(conf["classifier"]["ff_hid"], conf["classifier"]["num_out_classes"], bias=True)
#         )
        
        self.loss = F.cross_entropy