# Formal execution dependency closure
run.py -> exp.exp_pyramid_v2.Exp_Pyramid_V2 -> unchanged exp.exp_pyramid.Exp_Pyramid
-> exp.exp_main.Exp_Main -> exp.exp_basic.Exp_Basic.

Formal registry: models.PMBI_Net.Model -> models.TQNet.Model;
helper imports: utils.pmbi_helpers.preserve_rng / mlp (original implementations).
PMBI_Net contains its full constructor, MTFE/HBI/phase/readout and forward.
It never calls or inherits TQNet_Pyramid_V2.Model.
Legacy registry: models.TQNet_Pyramid_V2.Model retains its original dependency on
models.TQNet_Pyramid for numerical comparison only.

Data and optimizer: original data_provider -> data_loader -> utils.timefeatures /
StandardScaler; Adam, OneCycleLR initialization, type3 updates, validation and
full-test evaluator remain in the unchanged inherited implementation.
Explicit old-checkpoint evaluation: exp.checkpoint_compat validates provenance and
isolates outputs before calling the same evaluate_best method. It cannot train.
