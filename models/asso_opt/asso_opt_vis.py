import torchmetrics
import wandb
import torch as th
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

import cv2
cv2.setNumThreads(1)
th.set_num_threads(1)


class AssoConcept(pl.LightningModule):
    def init_weight_concept(self, concept2cls):
        self.init_weight = th.zeros((self.cfg.num_cls, len(self.select_idx))) #init with the actual number of selected index

        if self.cfg.use_rand_init: th.nn.init.kaiming_normal_(self.init_weight)
        else: self.init_weight.scatter_(0, concept2cls, self.cfg.init_val)
            
        if 'cls_name_init' in self.cfg and self.cfg.cls_name_init != 'none':
            if self.cfg.cls_name_init == 'replace':
                self.init_weight = th.load(self.init_weight_save_dir)
            elif self.cfg.cls_name_init == 'combine':
                self.init_weight += th.load(self.init_weight_save_dir)
                self.init_weight = self.init_weight.clip(max=1)
            elif self.cfg.cls_name_init == 'random':
                th.nn.init.kaiming_normal_(self.init_weight)


    def __init__(self, cfg, init_weight=None, select_idx=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        concept_feat_path = self.data_root.joinpath('concepts_feat_{}.pth'.format(self.cfg.clip_model.replace('/','-')))
        concept_raw_path = self.data_root.joinpath('concepts_raw_selected.npy')
        concept2cls_path = self.data_root.joinpath('concept2cls_selected.npy')
        select_idx_path = self.data_root.joinpath('select_idx.pth')
        self.init_weight_save_dir = self.data_root.joinpath('init_weight.pth')
        cls_sim_path = self.data_root.joinpath('cls_sim.pth')

        if not concept_feat_path.exists():
            raise RuntimeError('need to call datamodule precompute_txt before using the model')
        else:
            if select_idx is None: self.select_idx = th.load(select_idx_path)[:cfg.num_concept]
            else: self.select_idx = select_idx

            self.concepts = th.load(concept_feat_path)[self.select_idx].cuda()
            if self.cfg.use_txt_norm: self.concepts = self.concepts / self.concepts.norm(dim=-1, keepdim=True)

            #self.concept_raw = np.load(concept_raw_path)[self.select_idx]
            self.concept2cls = th.from_numpy(np.load(concept2cls_path)[self.select_idx]).long()#.view(1, -1)

        if init_weight is None:
            self.init_weight_concept(self.concept2cls)
        else:
            self.init_weight = init_weight

        if 'cls_sim_prior' in self.cfg and self.cfg.cls_sim_prior and self.cfg.cls_sim_prior != 'none':
            # class similarity is prior to restrict class-concept association
            # if class A and B are dissimilar (similarity=0), then the mask location will be 0 
            print('use cls prior')
            cls_sim = th.load(cls_sim_path)
            new_weights = []
            for concept_id in range(self.init_weight.shape[1]):
                target_class = int(th.where(self.init_weight[:,concept_id] == 1)[0])
                new_weights.append(cls_sim[target_class] + self.init_weight[:,concept_id])
            self.init_weight = th.vstack(new_weights).T
            # self.weight_mask = cls_sim @ self.init_weight

        self.asso_mat = th.nn.Parameter(self.init_weight.clone())


        ##for only fc
        self.visual2concept = th.nn.Linear(512, len(self.select_idx))


        self.kl_weight = self.cfg.init_kl
        
        #if self.cfg.concept2cls_loss:
        self.concept2cls = self.concept2cls.squeeze()
        if len(self.concept2cls.shape)>1:
            self.conceptwcls = self.concept2cls.t()
        else:
            self.conceptwcls = th.zeros((self.cfg.num_cls, len(self.select_idx)))
            self.conceptwcls.scatter_(0, self.concept2cls.unsqueeze(0), self.cfg.init_val)


        ### adapter 2
        self.adapter2 = th.nn.Linear(len(self.select_idx),len(self.select_idx))

        ###for disc
        self.W = th.nn.Parameter(th.Tensor(512, len(self.select_idx)))
        self.bias = th.nn.Parameter(th.Tensor(len(self.select_idx)))
        
        self.linear_logsigma_head1 = th.nn.Linear(512,512)
        self.linear_mean_head1 = th.nn.Linear(512,512)
       
        ### for sigmoid loss
        self.t_prime = th.nn.Parameter(th.tensor(np.log(10)))  # log 10
        self.b = th.nn.Parameter(th.tensor(-10.0))
        self.n_samples_inference = 4096


        self.register_buffer('temp', th.tensor(0.05))
        self.register_buffer('temptest', th.tensor(.005))
        priors = [0.0001]
        self.register_buffer('prior', th.tensor(priors))

        th.nn.init.xavier_normal_(self.W)
        self.bias.data.fill_(0.0)

        self.dropout = th.nn.Dropout(p=0.1)

 
 

        self.train_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls)
        self.valid_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls)
        # self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls, average='macro')
        self.test_acc = torchmetrics.Accuracy(num_classes=cfg.num_cls)
        self.all_y = []
        self.all_pred = []
        self.confmat = torchmetrics.ConfusionMatrix(self.cfg.num_cls)
        self.save_hyperparameters()


    def _get_weight_mat(self):
        if self.cfg.asso_act == 'relu':
            mat = F.relu(self.asso_mat)
        elif self.cfg.asso_act == 'tanh':
            mat = F.tanh(self.asso_mat) 
        elif self.cfg.asso_act == 'softmax':
            mat = F.softmax(self.asso_mat, dim=-1) 
        else:
            mat = self.asso_mat
        return mat 


    def forward(self, img_feat):
        mat = self._get_weight_mat()
        self.concepts = self.concepts.cpu()
        cls_feat = mat @ self.concepts
        sim = img_feat @ cls_feat.t()
        return sim


    def training_step(self, train_batch, batch_idx):
        image, label = train_batch

        sim = self.forward(image)
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        self.log('training_loss', cls_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        final_loss = cls_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss


    def configure_optimizers(self):
        opt = th.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return opt


    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        sim = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        self.log('val_loss', loss)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        image, y = batch
        sim = self.forward(image)
        conceptdis = image @ self.concepts.t()
        # recall_at_1 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(), k=1)
        # recall_at_2 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(),k=2)
        # recall_at_3 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(),k=3)
        # recall_at_5 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(),k=5)
        # recall_at_10 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(),k=10)
        # # 将结果记录到TensorBoard或其他地方
        # self.log('test_recall@1', recall_at_1)
        # self.log('test_recall@2', recall_at_2)
        # self.log('test_recall@3', recall_at_3)
        # self.log('test_recall@5', recall_at_5)
        # self.log('test_recall@10', recall_at_10)

        # recall_at_1 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(), k=0.9)
        # recall_at_2 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(),k=0.7)
        # recall_at_10 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(),k=0.5)
        # # 将结果记录到TensorBoard或其他地方
        # self.log('test_recall@0.9', recall_at_1)
        # self.log('test_recall@0.7', recall_at_2)
        # self.log('test_recall@0.5', recall_at_10)





        pred = 100 * sim
        #loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        # self.confmat(y_pred, y)
        # self.all_y.append(y)
        # self.all_pred.append(y_pred)
        # self.log('test_loss', loss)
        # self.test_acc(pred, y)
        # self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return y_pred, pred, conceptdis

    def compute_recall(self, match_scores, labels, gt_matrix, k):

        recalls = []
        for scores, gt in zip(match_scores,labels):

            true_positives = 0
            false_negatives = 0
            # 获取所有的索引
            all_indices = th.arange(scores.size(0)).cuda()
     
            # # 对匹配得分进行排序并获取top-k的索引
            # top_k_indices = th.argsort(scores, descending=True)[:k]

            # Min-Max 归一化
            min_value = th.min(scores)
            max_value = th.max(scores)
            min_max_normalized_scores = (scores - min_value) / (max_value - min_value)

            # 获取所有的索引
            all_indices = th.arange(min_max_normalized_scores.size(0)).cuda()
     
            # # 对匹配得分进行排序并获取top-k的索引
            # top_k_indices = th.argsort(scores, descending=True)[:k]
            # 选择大于25的索引
            top_k_indices = th.where(min_max_normalized_scores > k)[0]
         
            print(len(top_k_indices))
            # 获取未被top-k索引到的索引
            #missed_indices = th.setdiff1d(all_indices, top_k_indices)
            missed_indices = th.where(~th.isin(all_indices, top_k_indices))[0]

            #for row_indices in top_k_indices:
            # Check if any index in the row is related to the image_gt
            is_positive = any(gt_matrix[row_index, gt] for row_index in top_k_indices)
            
            if is_positive:
                true_positives += 1
            
            is_falsenegative = any(gt_matrix[row_index, gt] for row_index in missed_indices)
            if is_falsenegative:
                false_negatives += 1


            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            # # 获取top-k的类别
            # top_k_classes = [gt_row[idx] for idx in top_k_indices]
            # # 计算检索正确的数量
            # correct_count = sum(1 for cls in top_k_classes if cls == 1)
            # # 计算Recall @k
            # recall = correct_count / sum(gt_row == 1)
            recalls.append(recall)
            recalls = [float(recall) for recall in recalls]
        return th.mean(th.tensor(recalls))



    def test_epoch_end(self, outputs):
        all_y = th.hstack(self.all_y)
        all_pred = th.hstack(self.all_pred)
        self.total_test_acc = self.test_acc(all_pred, all_y)
        pass


    def on_predict_epoch_start(self):
        self.num_pred = 4
        self.concepts = self.concepts.to(self.device)

        self.pred_table = wandb.Table(
            columns=["img", "label"] +
            ["pred_{}".format(i) for i in range(self.num_pred)])


    def predict_step(self, batch, batch_idx):
        image, y, image_name = batch
        sim = self.forward(image)
        pred = 100 * sim
        _, y_pred = th.topk(pred, self.num_pred)
        for img_path, gt, top_pred in zip(image_name, y, y_pred):
            gt = (gt + 1).item()
            top_pred = (top_pred + 1).tolist()
            self.pred_table.add_data(wandb.Image(img_path), gt, *top_pred)
    

    def on_predict_epoch_end(self, results):
        wandb.log({'pred_table': self.pred_table})
    

    def prune_asso_mat(self, q=0.9, thresh=None):
        asso_mat = self._get_weight_mat().detach()
        val_asso_mat = th.abs(asso_mat).max(dim=0)[0]
        if thresh is None:
            thresh = th.quantile(val_asso_mat, q)
        good = val_asso_mat >= thresh
        return good


    def extract_cls2concept(self, cls_names, thresh=0.05):
        asso_mat = self._get_weight_mat().detach()
        strong_asso = asso_mat > thresh 
        res = {}
        import pdb; pdb.set_trace()
        for i, cls_name in enumerate(cls_names): 
            ## threshold globally
            keep_idx = strong_asso[i]
            ## sort
            res[cls_name] = np.unique(self.concept_raw[keep_idx])
        return res


    def extract_concept2cls(self, percent_thresh=0.95, mode='global'):
        asso_mat = self.asso_mat.detach()
        res = {} 
        for i in range(asso_mat.shape[1]):
            res[i] = th.argsort(asso_mat[:, i], descending=True).tolist()
        return res

def bernoulli_kl(p, q, eps=1e-7):
    return (p * ((p + eps).log() - (q + eps).log())) + (1. - p) * ((1. - p + eps).log() - (1. - q + eps).log())

 

class AssoConceptFast(AssoConcept):

    def forward(self, dot_product):
        mat = self._get_weight_mat()
        return dot_product @ mat.t()
    

class AssoConceptfc(AssoConcept):

    def forward(self, img_feat):
        mat = self._get_weight_mat()
        
        new_conceptdis = self.visual2concept(img_feat)
 
        dot_product = img_feat @ self.concepts.t()
         
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product 
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product = self.forward(image)
       
        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    

        if self.cfg.kl_cos1: #direct klloss weight cos dicline
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
        else: #direct klloss weight cos dicline every n epoch
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product  = self.forward(image)
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance
 

class AssoConceptfcc2c(AssoConcept):

    def forward(self, img_feat):
        mat = self._get_weight_mat()
        #print(img_feat.shape)
        new_conceptdis = self.visual2concept(img_feat)
 
        dot_product = img_feat @ self.concepts.t()
         
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product 
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product = self.forward(image)
       
        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        
        
        
        
        
        
        
        
        
        
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              


        
        
        
        
        
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        # if self.cfg.kl_cos1:
        #     self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        # else:
        #     if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
        #         if self.cfg.kl_cos:
        #             self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
        #         else:
        #             self.kl_weight *= self.cfg.kl_decay_factor
      
        ### only for 32
        
        # 仅在指定的epochs间隔达到后才进行退火
        if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0:
            # 检查是否使用余弦退火
            if self.cfg.kl_cos:
                # 计算新的KL权重
                self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
            else:
                # 使用简单衰减的方式更新KL权重
                self.kl_weight *= self.cfg.kl_decay_factor
            # 确保权重不低于最小值
            self.kl_weight = max(self.kl_weight, self.cfg.kl_min)



        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product  = self.forward(image)
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance
 
    

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()


class AssoConceptfcc2c_z(AssoConcept):

    def forward(self, img_feat):
        mat = self._get_weight_mat()
         
        dot_product = img_feat @ self.concepts.t()

        logits = th.nn.functional.linear(img_feat, self.W.T, self.bias)
        logits = self.dropout(logits)
        mask = self.bin_concrete_sample(logits, self.temp).clamp(1e-6, 1.-1e-6)
        kl_z = bernoulli_kl(th.sigmoid(logits), self.prior).sum(1).mean()
          
        #sparse
        new_conceptdis = dot_product * mask

        return new_conceptdis @ mat.t(), new_conceptdis, dot_product, kl_z
    
    

    def val_forward(self, img_feat):
        mat = self._get_weight_mat()
         
        dot_product = img_feat @ self.concepts.t()

        logits = th.nn.functional.linear(img_feat, self.W.T, self.bias)
        mask = RelaxedBernoulli(self.temptest, logits=logits).sample()
        kl_z = bernoulli_kl(th.sigmoid(logits), self.prior).sum(1).mean()

        # mask = self.bin_concrete_sample(logits, self.temp).clamp(1e-6, 1.-1e-6)
        # kl_z = bernoulli_kl(th.sigmoid(logits), self.prior).sum(1).mean()

        #sparse
        new_conceptdis = dot_product * mask

        return new_conceptdis @ mat.t(), new_conceptdis, dot_product, kl_z
    
    def bin_concrete_sample(self, a, temperature, eps=1e-8):
        """"
        Sample from the binary concrete distribution
        """

        U = th.rand_like(a).clamp(eps, 1. - eps)
        L = th.log(U) - th.log(1. - U)
        X = th.sigmoid((L + a) / temperature)


    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product, kl_z = self.forward(image)
       
        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        
        
        
        
        
        
        
        
        
        
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              


        
        
        
        
        
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
        kl_z_loss = self.cfg.sparse*kl_z
        self.log('kl_z_loss', kl_z_loss)
        final_loss = final_loss + kl_z_loss




        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product, kl_z  = self.val_forward(image)
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        self.log('kl_z',kl_z)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product = self.val_forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance
 
    

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()

    def bin_concrete_sample(self, a, temperature, eps=1e-8):
        """"
        Sample from the binary concrete distribution
        """

        U = th.rand_like(a).clamp(eps, 1. - eps)
        L = th.log(U) - th.log(1. - U)
        X = th.sigmoid((L + a) / temperature)

        return X


class AssoConceptfcc2c_z_add(AssoConcept):

    def forward(self, img_feat):
        mat = self._get_weight_mat()
         
        dot_product = img_feat @ self.concepts.t()
        ###adapter
        conceptdis = self.visual2concept(img_feat)
        new_conceptdis = dot_product + conceptdis
        
        logits = th.nn.functional.linear(img_feat, self.W.T, self.bias)
        mask = self.bin_concrete_sample(logits, self.temp).clamp(1e-6, 1.-1e-6)
        kl_z = bernoulli_kl(th.sigmoid(logits), self.prior).sum(1).mean()
          
        #sparse
        new_conceptdis = new_conceptdis * mask

        return new_conceptdis @ mat.t(), new_conceptdis, dot_product, kl_z
    
    
    def val_forward(self, img_feat):
        mat = self._get_weight_mat()
         
        dot_product = img_feat @ self.concepts.t()

        logits = th.nn.functional.linear(img_feat, self.W.T, self.bias)
        mask = RelaxedBernoulli(self.temptest, logits=logits).sample()
        kl_z = bernoulli_kl(th.sigmoid(logits), self.prior).sum(1).mean()

        # mask = self.bin_concrete_sample(logits, self.temp).clamp(1e-6, 1.-1e-6)
        # kl_z = bernoulli_kl(th.sigmoid(logits), self.prior).sum(1).mean()

        #sparse
        new_conceptdis = dot_product * mask

        return new_conceptdis @ mat.t(), new_conceptdis, dot_product, kl_z
    
    def bin_concrete_sample(self, a, temperature, eps=1e-8):
        """"
        Sample from the binary concrete distribution
        """

        U = th.rand_like(a).clamp(eps, 1. - eps)
        L = th.log(U) - th.log(1. - U)
        X = th.sigmoid((L + a) / temperature)


    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product, kl_z = self.forward(image)
       
        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        
        
        
        
        
        
        
        
        
        
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              


        
        
        
        
        
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
        kl_z_loss = self.cfg.sparse*kl_z
        self.log('kl_z_loss', kl_z_loss)
        final_loss = final_loss + kl_z_loss




        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product, kl_z  = self.val_forward(image)
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        self.log('kl_z',kl_z)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product = self.val_forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance
 
    

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()

    def bin_concrete_sample(self, a, temperature, eps=1e-8):
        """"
        Sample from the binary concrete distribution
        """

        U = th.rand_like(a).clamp(eps, 1. - eps)
        L = th.log(U) - th.log(1. - U)
        X = th.sigmoid((L + a) / temperature)

        return X



class AssoConceptfcc2c_add(AssoConcept):

    def forward(self, img_feat):
        mat = self._get_weight_mat()
        
        dot_product = img_feat @ self.concepts.t()     
        
        # ###patch_pool
        # side_logits = img_feat_patch @ self.concepts.t() #B,N,N_c
        # side_logits = th.mean(side_logits, dim=1).squeeze()
        # ###adapter
        conceptdis = self.visual2concept(img_feat)
        #conceptdis = self.adapter2(side_logits)
        
        new_conceptdis = dot_product + self.cfg.adapter_weight*conceptdis
         
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product 
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product = self.forward(image)
       
        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
         
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              
   
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product  = self.forward(image)
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance  

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()

 

class AssoConceptfcc2c_add_2(AssoConcept):

    def forward(self, img_feat):
        mat = self._get_weight_mat()
        
        img_feat_cls = img_feat[:,0,:]
        img_feat_patch = img_feat[:,1:,:]
        
        dot_product = img_feat_cls @ self.concepts.t()     
        
        ###patch_pool
        side_logits = img_feat_patch @ self.concepts.t() #B,N,N_c
        side_logits = th.mean(side_logits, dim=1).squeeze()
        ###adapter
        #conceptdis = self.visual2concept(side_logits)
        conceptdis = self.adapter2(side_logits)
        
        new_conceptdis = dot_product + self.cfg.adapter_weight*conceptdis
         
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product 
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product = self.forward(image)
       
        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
         
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              


        
        
        
        
        
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product  = self.forward(image)
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance
 
    

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()


class AssoConceptfcc2c_add_3(AssoConcept):

    def forward(self, img_feat):
        mat = self._get_weight_mat()
        
        img_feat_cls = img_feat[:,0,:]
        img_feat_patch = img_feat[:,1:,:]
        
        dot_product = img_feat_cls @ self.concepts.t()     
        
        ###patch_pool
         
        side_logits = th.mean(img_feat_patch, dim=1).squeeze() #b,n,c --- b.1.c
        ###adapter
        conceptdis = self.visual2concept(side_logits)
        #conceptdis = self.adapter2(side_logits)
        
        new_conceptdis = dot_product + self.cfg.adapter_weight*conceptdis
         
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product 
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product = self.forward(image)
       
        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
         
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              


        
        
        
        
        
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product  = self.forward(image)
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance
 
    

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()

def sample_gaussian_tensors(mu, logsigma, num_samples):
    # 生成标准正态分布的随机数
    eps = th.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)
    
    # 将标准差扩展维度，使其形状变为 (num_concepts, 1, feature_dim)
    sigma_expanded = th.exp(logsigma.unsqueeze(1))

    # 将随机数乘以扩展后的标准差，并加上均值
    samples = eps.mul(sigma_expanded).add_(mu.unsqueeze(1))
    
    return samples.mean(dim=1).squeeze()

class AssoConceptfcc2c_prob_dot(AssoConcept):

    def get_uncertainty(self, pred_concept_logsigma):
        uncertainty = pred_concept_logsigma.mean(dim=-1).exp()
        return uncertainty
    
    def forward(self, img_feat):
        mat = self._get_weight_mat()
        
        dot_product = img_feat @ self.concepts.t()     
        
        # ###patch_pool
        # side_logits = img_feat_patch @ self.concepts.t() #B,N,N_c
        # side_logits = th.mean(side_logits, dim=1).squeeze()
        # ###adapter
        pred_concept_mean = img_feat
        pred_concept_logsigma = self.linear_logsigma_head1(img_feat)
        pred_concept_logsigma = th.clip(pred_concept_logsigma, max=10)
        pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference) # B x num_concepts x n_samples x hidden_dim
        concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)
        #conceptdis = self.visual2concept(img_feat)
        
        #pred_concept_logsigma = self.logsigma_head(self.concepts.permute(1, 0)).view(self.cfg.num_concept, -1)
        #conceptdis = self.adapter2(side_logits)
        
        #new_conceptdis = dot_product + self.cfg.adapter_weight*conceptdis
        new_conceptdis = pred_embeddings @ self.concepts.t()  
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product, concept_uncertainty
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        concept_uncertainty = concept_uncertainty.mean()

        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
         
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              
   
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        self.log('uncertainty', concept_uncertainty)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty  = self.forward(image)
        concept_uncertainty = concept_uncertainty.mean()
        
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        self.log('uncertainty',concept_uncertainty)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance  

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()


def compute_covariance(features):
    """
    Compute the covariance matrix for a given set of features.
    """
    n = features.size(0)
    mean = th.mean(features, dim=0, keepdim=True)
    features_centered = features - mean
    cov = (features_centered.T @ features_centered) / (n - 1)
    return cov + 1e-6 * th.eye(features.size(1), device=features.device)   # Regularization for numerical stability

def mahalanobis_distance_squared(x, mu, cov_inv):
    """
    Compute the squared Mahalanobis distance.
    """
    delta = x - mu
    return (delta @ cov_inv @ delta.t()).diagonal()

def mahalanobis_distance_batch(text_features, image_features, cov_inv):
    """
    Compute the Mahalanobis distance in a batched manner.
    """
    delta = text_features.unsqueeze(1) - image_features  # Broadcasting difference
    m_distances = th.einsum('bik,kj,bij->bi', delta, cov_inv, delta)  # Batch matrix multiplication
    return m_distances



class AssoConceptfcc2c_prob_ma_1(AssoConcept):

    def get_uncertainty(self, pred_concept_logsigma):
        uncertainty = pred_concept_logsigma.mean(dim=-1).exp()
        return uncertainty
    
    def forward(self, img_feat):
        mat = self._get_weight_mat()
      
      
        self.concepts = self.concepts.cpu()

        dot_product = img_feat @ self.concepts.t()  
        
        # ###patch_pool
        # side_logits = img_feat_patch @ self.concepts.t() #B,N,N_c
        # side_logits = th.mean(side_logits, dim=1).squeeze()
        # ###adapter
        pred_concept_mean = self.linear_mean_head1(img_feat)
        pred_concept_logsigma = self.linear_logsigma_head1(img_feat)

        pred_concept_logsigma = th.clip(pred_concept_logsigma, max=10)
        pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)
        
        
        pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference) # B x num_concepts x n_samples x hidden_dim
        concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)
        #conceptdis = self.visual2concept(img_feat)
        
        #pred_concept_logsigma = self.logsigma_head(self.concepts.permute(1, 0)).view(self.cfg.num_concept, -1)
        #conceptdis = self.adapter2(side_logits)
        
        #new_conceptdis = dot_product + self.cfg.adapter_weight*conceptdis
        new_conceptdis = pred_embeddings @ self.concepts.t()  
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product, concept_uncertainty, pred_embeddings
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product, concept_uncertainty, pred_embeddings = self.forward(image)
        concept_uncertainty = concept_uncertainty.mean()

        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
         
        ###cal_embedding_mu
        class_embedding = th.zeros((self.cfg.num_cls, 512)).cuda()
        label_counts_1 = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_embedding.index_add_(0, label, pred_embeddings)
        label_counts_1.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts_1[label_counts_1 == 0] = 1
        # 计算均值
        pred_embeddings_mu  = class_embedding / label_counts_1.view(-1, 1)

   
        if self.cfg.ma_distance:
            # Compute the covariance matrix for the image features
            image_cov = compute_covariance(pred_embeddings_mu)
            image_cov_inv = th.inverse(image_cov)
            # Compute the Mahalanobis distance for all pairs of image and text features
            distances = mahalanobis_distance_batch(self.concepts, pred_embeddings_mu, image_cov_inv)
 
      
        if self.cfg.concept2cls_loss:
     
 
                if self.cfg.concept2cls_12all:
                    if self.cfg.ma_distance:
                    #if True:
                        # Compute the covariance matrix for the image features
                        image_cov = compute_covariance(pred_embeddings_mu)
                        image_cov_inv = th.inverse(image_cov)
                        # Compute the Mahalanobis distance for all pairs of image and text features
                        distances = mahalanobis_distance_batch(self.concepts, pred_embeddings_mu, image_cov_inv)
                        c2c_loss = self.customContraLoss(distances.t(), self.conceptwcls.cuda(), tau=0.07)
                    else:
                        #print(self.conceptwcls.shape)
                        c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                else:
                    print('Nothing for num_attri<num_cls')
              
   
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        self.log('uncertainty', concept_uncertainty)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty,_  = self.forward(image)
        concept_uncertainty = concept_uncertainty.mean()
        
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        self.log('uncertainty',concept_uncertainty)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty, img_feat = self.forward(image)

        
        # recall_at_1 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(), k=0.9)
        # recall_at_2 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(),k=0.7)
        # recall_at_10 = self.compute_recall(conceptdis, y, self.conceptwcls.cuda().t(),k=0.5)
        # # 将结果记录到TensorBoard或其他地方
        # self.log('test_recall@0.9', recall_at_1)
        # self.log('test_recall@0.7', recall_at_2)
        # self.log('test_recall@0.5', recall_at_10)


        pred = 100 * sim
        #loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        # self.confmat(y_pred, y)
        # self.all_y.append(y)
        # self.all_pred.append(y_pred)
        # self.log('test_loss', loss)
        # self.log('concept_uncertainty',concept_uncertainty)
        # self.test_acc(pred, y)
        # self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        
        return y_pred, pred, conceptdis, concept_uncertainty
        # when want to anaylze
        #return y_pred, conceptdis, dot_product

    
    def compute_recall(self, match_scores, labels, gt_matrix, k):

        recalls = []
        for scores, gt in zip(match_scores,labels):

            true_positives = 0
            false_negatives = 0
            # 获取所有的索引
            all_indices = th.arange(scores.size(0)).cuda()
     
            # # 对匹配得分进行排序并获取top-k的索引
            # top_k_indices = th.argsort(scores, descending=True)[:k]

            # Min-Max 归一化
            min_value = th.min(scores)
            max_value = th.max(scores)
            min_max_normalized_scores = (scores - min_value) / (max_value - min_value)

            # 获取所有的索引
            all_indices = th.arange(min_max_normalized_scores.size(0)).cuda()
     
            # # 对匹配得分进行排序并获取top-k的索引
            # top_k_indices = th.argsort(scores, descending=True)[:k]
            # 选择大于25的索引
            top_k_indices = th.where(min_max_normalized_scores > k)[0]
         
            print(len(top_k_indices))
            # 获取未被top-k索引到的索引
            #missed_indices = th.setdiff1d(all_indices, top_k_indices)
            missed_indices = th.where(~th.isin(all_indices, top_k_indices))[0]

            #for row_indices in top_k_indices:
            # Check if any index in the row is related to the image_gt
            is_positive = any(gt_matrix[row_index, gt] for row_index in top_k_indices)
            
            if is_positive:
                true_positives += 1
            
            is_falsenegative = any(gt_matrix[row_index, gt] for row_index in missed_indices)
            if is_falsenegative:
                false_negatives += 1


            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            # # 获取top-k的类别
            # top_k_classes = [gt_row[idx] for idx in top_k_indices]
            # # 计算检索正确的数量
            # correct_count = sum(1 for cls in top_k_classes if cls == 1)
            # # 计算Recall @k
            # recall = correct_count / sum(gt_row == 1)
            recalls.append(recall)
            recalls = [float(recall) for recall in recalls]
        return th.mean(th.tensor(recalls))
  
     
    def compute_recall_topnk(self, match_scores, labels, gt_matrix, n, k):

        recalls = []
        for scores, gt in zip(match_scores,labels):

            true_positives = 0
            false_negatives = 0
            # 获取所有的索引
            all_indices = th.arange(scores.size(0)).cuda()
     
            # # 对匹配得分进行排序并获取top-k的索引
            # top_k_indices = th.argsort(scores, descending=True)[:k]

            # Min-Max 归一化
            min_value = th.min(scores)
            max_value = th.max(scores)
            min_max_normalized_scores = (scores - min_value) / (max_value - min_value)

            # 获取所有的索引
            all_indices = th.arange(min_max_normalized_scores.size(0)).cuda()
     
            # # 对匹配得分进行排序并获取top-k的索引
            # top_k_indices = th.argsort(scores, descending=True)[:k]
            # 选择大于25的索引
            top_k_indices = th.where(min_max_normalized_scores > n)[0]
         
            top_n_indices = top_k_indices[th.argsort(scores[top_k_indices], descending=True)[:k]]

            print(len(top_n_indices))
            # 获取未被top-k索引到的索引
            #missed_indices = th.setdiff1d(all_indices, top_k_indices)
            missed_indices = th.where(~th.isin(all_indices, top_k_indices))[0]

            #for row_indices in top_k_indices:
            # Check if any index in the row is related to the image_gt
            is_positive = any(gt_matrix[row_index, gt] for row_index in top_k_indices)
            
            if is_positive:
                true_positives += 1
            
            is_falsenegative = any(gt_matrix[row_index, gt] for row_index in missed_indices)
            if is_falsenegative:
                false_negatives += 1


            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            # # 获取top-k的类别
            # top_k_classes = [gt_row[idx] for idx in top_k_indices]
            # # 计算检索正确的数量
            # correct_count = sum(1 for cls in top_k_classes if cls == 1)
            # # 计算Recall @k
            # recall = correct_count / sum(gt_row == 1)
            recalls.append(recall)
        return th.mean(th.tensor(recalls))
  
    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
     
    
    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()
 
   
# The rest of the InfoNCE loss calculation remains the same.


class AssoConceptfcc2c_prob_dot_1(AssoConcept):

    def get_uncertainty(self, pred_concept_logsigma):
        uncertainty = pred_concept_logsigma.mean(dim=-1).exp()
        return uncertainty
    
    def forward(self, img_feat):
        mat = self._get_weight_mat()
        
        dot_product = img_feat @ self.concepts.t()     
        
        # ###patch_pool
        # side_logits = img_feat_patch @ self.concepts.t() #B,N,N_c
        # side_logits = th.mean(side_logits, dim=1).squeeze()
        # ###adapter
        pred_concept_mean = self.linear_mean_head1(img_feat)
        pred_concept_logsigma = self.linear_logsigma_head1(img_feat)

        pred_concept_logsigma = th.clip(pred_concept_logsigma, max=10)
        pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)
        
        
        pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference) # B x num_concepts x n_samples x hidden_dim
        concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)
        #conceptdis = self.visual2concept(img_feat)
        
        #pred_concept_logsigma = self.logsigma_head(self.concepts.permute(1, 0)).view(self.cfg.num_concept, -1)
        #conceptdis = self.adapter2(side_logits)
        
        #new_conceptdis = dot_product + self.cfg.adapter_weight*conceptdis
        new_conceptdis = pred_embeddings @ self.concepts.t()  
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product, concept_uncertainty
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        concept_uncertainty = concept_uncertainty.mean()

        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
         
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

      
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              
   
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        self.log('uncertainty', concept_uncertainty)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty  = self.forward(image)
        concept_uncertainty = concept_uncertainty.mean()
        
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        self.log('uncertainty',concept_uncertainty)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance  

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()


class AssoConceptfcc2c_add_dot(AssoConcept):

    def get_uncertainty(self, pred_concept_logsigma):
        uncertainty = pred_concept_logsigma.mean(dim=-1).exp()
        return uncertainty
    
    def forward(self, img_feat):
        mat = self._get_weight_mat()
        
        dot_product = img_feat @ self.concepts.t()     
        
        # ###patch_pool
        # side_logits = img_feat_patch @ self.concepts.t() #B,N,N_c
        # side_logits = th.mean(side_logits, dim=1).squeeze()
        # ###adapter
        pred_concept_mean = img_feat
        
        adapter_ab = self.linear_logsigma_head1(img_feat)
        # pred_concept_logsigma = th.clip(pred_concept_logsigma, max=10)
        # pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference) # B x num_concepts x n_samples x hidden_dim
        # concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)
        #conceptdis = self.visual2concept(img_feat)
        
        #pred_concept_logsigma = self.logsigma_head(self.concepts.permute(1, 0)).view(self.cfg.num_concept, -1)
        #conceptdis = self.adapter2(side_logits)
        
        #new_conceptdis = dot_product + self.cfg.adapter_weight*conceptdis
        new_conceptdis = adapter_ab @ self.concepts.t()  
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product 
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product = self.forward(image)
        #concept_uncertainty = concept_uncertainty.mean()

        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
         
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              
   
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        #self.log('uncertainty', concept_uncertainty)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product  = self.forward(image)
        #concept_uncertainty = concept_uncertainty.mean()
        
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        #self.log('uncertainty',concept_uncertainty)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance  

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()

 





class AssoConceptfcc2c_dot(AssoConcept):

    def forward(self, img_feat):
        mat = self._get_weight_mat()
        
        dot_product = img_feat @ self.concepts.t()        
        conceptdis = self.visual2concept(img_feat)

        new_conceptdis = dot_product * conceptdis
         
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product 
    
    
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product = self.forward(image)
       
        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        
        
        
        
        
        
        
        
        
        
        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              


        
        
        
        
        
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        #self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product  = self.forward(image)
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        #self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance
 
    

    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()


       

# 12. cross 都学

class AssoConcept_woproblistic_862_wcross12_decaykl_c2c(AssoConcept):

    def generate_soft_labels(self, y_true, smooth_param=0.1):
        y_true = th.tensor(y_true, dtype=th.float32)
        soft_labels = (1 - smooth_param) * y_true + smooth_param / y_true.size(1)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        return soft_labels

    def get_uncertainty(self, pred_concept_logsigma):
        uncertainty = pred_concept_logsigma.mean(dim=-1).exp()
        return uncertainty
    
    def forward(self, img_feat):
        mat = self._get_weight_mat()
        B = img_feat.shape[0]
        #cls_feat = mat @ self.concepts
        # if len(self.concepts.shape)<3:
        #     self.concepts = self.concepts.unsqueeze(0).repeat(B, 1, 1)
        img_feat_cls_org = img_feat[:,0,:]
        
        img_feat = self.pos_drop(img_feat + self.pos_embed)
        if len(self.concepts.shape)<3:
            self.concepts = self.concepts.unsqueeze(0).repeat(B, 1, 1)

        output = self.attn(q=self.norm1(img_feat),
                           k=self.norm2(self.concepts),
                           v=self.norm2(self.concepts),
                           )

        
        # output = self.multihead_attn2(q=self.norm1(img_feat),
        #                    k=self.norm2(self.concepts),
        #                    v=self.norm2(self.concepts),
        #                    )
        
        output = img_feat + self.drop_path(output)
        
        feat_v_a = output[:,1:,:]
        feat_v_a = feat_v_a.mean(1) 
        
        img_feat_cls = img_feat_cls_org + feat_v_a
        # pred_concept_mean = self.linear_mean_head(img_feat_cls)#.view(B, self.num_concepts, -1)
        # pred_concept_logsigma = self.linear_logsigma_head(img_feat_cls)#.view(B, self.num_concepts, -1)
        # pred_concept_logsigma = th.clip(pred_concept_logsigma, max=10)
        # pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)
        
        # pred_embeddings = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference) # B x num_concepts x n_samples x hidden_dim
        # #concept_logit, concept_prob = self.match_prob(pred_embeddings, concept_embeddings)
        # concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)
        
        #conceptdis = self.visual2concept1(img_feat_cls)
        
        # pred_concept_mean = self.linear_mean_head_true(img_feat_cls)#.view(B, self.num_concepts, -1)
        # pred_concept_logsigma = self.linear_logsigma_head_true(img_feat_cls)#.view(B, self.num_concepts, -1)
        # pred_concept_logsigma = th.clip(pred_concept_logsigma, max=10)
        # pred_concept_mean = F.normalize(pred_concept_mean, p=2, dim=-1)
        
        # new_conceptdis = sample_gaussian_tensors(pred_concept_mean, pred_concept_logsigma, self.n_samples_inference) # B x num_concepts x n_samples x hidden_dim
        # #concept_logit, concept_prob = self.match_prob(pred_embeddings, concept_embeddings)
        # concept_uncertainty = self.get_uncertainty(pred_concept_logsigma)

        new_conceptdis = self.visual2concept(img_feat_cls)
        concept_uncertainty = 0


        if img_feat_cls_org.dtype == th.float16:
            img_feat_cls_org = img_feat_cls_org.to(th.float32)

        dot_product = img_feat_cls_org @ self.concepts[0,:,:].t()
         
        #sim = pred_embeddings @ self.concepts.t()
        return new_conceptdis @ mat.t(), new_conceptdis, dot_product, concept_uncertainty
 
    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
 
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
      

        #cal预测的矩阵 Nclass x N concept
        class_conceptdis  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        #class2concept[label] = img_feat_cls[label]
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        # 使用索引和聚合操作计算均值
        class_conceptdis.index_add_(0, label, conceptdis)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu  = class_conceptdis / label_counts.view(-1, 1)


        #callabel的矩阵 Nclass x N concept
        class_conceptdis_label  = th.zeros((self.cfg.num_cls, len(self.select_idx))).cuda()
        label_counts = th.zeros(self.cfg.num_cls).cuda()
        class_conceptdis_label.index_add_(0, label, dot_product)
        label_counts.index_add_(0, label, th.ones_like(label, dtype=th.float32))
        # 防止除以零，将计数中的零替换为1
        label_counts[label_counts == 0] = 1
        # 计算均值
        class2concept_mu_label  = class_conceptdis_label / label_counts.view(-1, 1)

 
        if self.cfg.concept2cls_loss:
     

                if self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_123 and self.cfg.concept2cls_12all:
                    self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss4(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,self.concept2clsall,tau=0.07)
                    
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122 and self.cfg.concept2cls_12all:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,:2]] = 1
                    self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss3(class2concept_mu,self.new_matrix1,self.new_matrix2,self.new_matrix3,tau=0.07)

 
                elif self.cfg.concept2cls_121 and self.cfg.concept2cls_122:
                    #self.concept2clsall = self.concept2cls

                    # 获取每行中最大的两个值及其索引
                    top_values, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    self.new_matrix1 = th.zeros_like(self.conceptwcls).cuda()
                    self.new_matrix2 = th.zeros_like(self.conceptwcls).cuda()
                    #self.new_matrix3 = th.zeros_like(self.conceptwcls).cuda()

                    # 在每行中最大的两个值的位置设置为1
                    self.new_matrix1[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices[:,0]] = 1
                    self.new_matrix2[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1
                    #self.new_matrix3[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    c2c_loss = self.customContraLoss2(class2concept_mu,self.new_matrix1,self.new_matrix2,tau=0.07)
              
                #num<num_cls
                elif self.cfg.concept2cls_12all and self.cfg.concept2cls_121:
                    self.conceptwcls1 = self.conceptwcls

                     # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1
                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls2 = new_matrix
                    c2c_loss = self.customContraLoss2(class2concept_mu, self.conceptwcls1.cuda(), self.conceptwcls2.cuda(), tau=0.07)


                elif self.cfg.concept2cls_12all:
                    #print(self.conceptwcls.shape)
                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)

                elif self.cfg.concept2cls_121:
                    # 检查每行是否全为零
                    is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)

                    # 对全为零的行进行处理
                    #max_indices = th.argmax(self.conceptwcls.cuda() * class2concept_mu_label, dim=1)

                    #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行

                    # Create a new matrix with zeros
                    new_matrix = th.zeros_like(self.conceptwcls)
                    # Set the maximum index position in each row to 1

                    # 仅为非 -1 的行进行赋值
                    valid_indices = max_indices != -1
                    new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    #new_matrix[th.arange(new_matrix.size(0)), max_indices] = 1
                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                elif self.cfg.concept2cls_122:

                    ### at this circumstance all row are not zero
                    # # 检查每行是否全为零
                    # is_all_zero_row = th.all(self.conceptwcls.cuda() * class2concept_mu_label == 0, dim=1)
                    # #concepts数目小于numclass的时候 判断是否一行对应的元素非零元素小于等于1 
                    # max_indices = th.argmax(self.conceptwcls.cuda()*class2concept_mu_label, dim=1)
                    # max_indices[is_all_zero_row] = -1  # 用特殊值标记全为零的行
                    # # Create a new matrix with zeros
                    # new_matrix = th.zeros_like(self.conceptwcls)
                    # # Set the maximum index position in each row to 1

                    # # 仅为非 -1 的行进行赋值
                    # valid_indices = max_indices != -1
                    # new_matrix[th.arange(new_matrix.size(0))[valid_indices], max_indices[valid_indices]] = 1

                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=2, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
                
                          
                elif self.cfg.concept2cls_123:

                    ### at this circumstance all row are not zero
                  
                    # 获取每行中最大的两个值及其索引
                    _, top_indices = th.topk(self.conceptwcls.cuda() * class2concept_mu_label, k=3, dim=1)

                    # 创建一个与 self.conceptwcls1 形状相同的零张量
                    self.new_matrix = th.zeros_like(self.conceptwcls)

                    # 在每行中最大的两个值的位置设置为1
                    new_matrix[th.arange(self.conceptwcls.size(0)).unsqueeze(1), top_indices] = 1

                    # the most similar concept
                    self.conceptwcls = new_matrix

                    c2c_loss = self.customContraLoss(class2concept_mu, self.conceptwcls.cuda(), tau=0.07)
       
                
                
                else:
                    print('Nothing for num_attri<num_cls')
              





        ### disloss
        # min-max normalize
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))

        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim  # scaling as standard CLIP does

        # classification accuracy
        cls_loss = F.cross_entropy(pred, label)
        if th.isnan(cls_loss):
            import pdb; pdb.set_trace() # yapf: disable

        # diverse response
        div = -th.var(sim, dim=0).mean()
        
        if self.cfg.asso_act == 'softmax':
            row_l1_norm = th.linalg.vector_norm(F.softmax(self.asso_mat,
                                                          dim=-1),
                                                ord=1,
                                                dim=-1).mean()
        # asso_mat sparse regulation
        row_l1_norm = th.linalg.vector_norm(self.asso_mat, ord=1,
                                            dim=-1).max() #.mean()
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('Learning Rate', current_lr)

        self.log('training_loss', cls_loss)
        self.log('dis_loss',dis_loss)
        if self.cfg.concept2cls_loss:
            self.log('c2c_loss1', c2c_loss)
        self.log('mean l1 norm', row_l1_norm)
        self.log('div', div)
        self.log('concept_uncertainty',concept_uncertainty)
        #print(concept_uncertainty)

        self.train_acc(pred, label)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        
        if self.cfg.concept2cls_loss:
            #final_loss = cls_loss + self.kl_weight*dis_loss
            final_loss = cls_loss + self.kl_weight*dis_loss + self.cfg.c2c_weight*c2c_loss
        else:
            final_loss = cls_loss + self.kl_weight*dis_loss #+ self.cfg.c2c_weight*c2c_loss
   
      
        #final_loss = cls_loss + 0.5*dis_loss
        if self.cfg.use_l1_loss:
            final_loss += self.cfg.lambda_l1 * row_l1_norm
        if self.cfg.use_div_loss:
            final_loss += self.cfg.lambda_div * div
        return final_loss

    def training_epoch_end(self, outputs):
        # 在每个 epoch 结束时调整 KL 散度的权重
    
        if self.cfg.kl_cos1:
            self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
      
 
        else:
            if (self.current_epoch + 1) % self.cfg.kl_decay_every == 0 and self.kl_weight > self.cfg.kl_min:
           
                if self.cfg.kl_cos:
                    self.kl_weight = self.cosine_annealing(self.current_epoch, self.cfg.max_epochs, self.cfg.init_kl, self.cfg.kl_min)
                else:
                    self.kl_weight *= self.cfg.kl_decay_factor

        self.log('kl_weight', self.kl_weight)
 
    def cosine_annealing(self, epoch, max_epochs, initial_value, final_value):
        epoch_tensor = th.tensor(epoch, dtype=th.float32)
        max_epochs_tensor = th.tensor(max_epochs, dtype=th.float32)
        
        cos_value = 0.5 * (1 + th.cos(th.pi * epoch_tensor / max_epochs_tensor))
        return final_value + 0.5 * (initial_value - final_value) * cos_value
            
    def validation_step(self, batch, batch_idx):
        if not self.cfg.DEBUG:
            if self.global_step == 0 and not self.cfg.DEBUG:
                wandb.define_metric('val_acc', summary='max')
        image, y = batch
        
        image = image.cuda() #if on_gpu else img_feat
        y = y.cuda() #if on_gpu else label
        
        #sim, concept_uncertainty = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        # dis_loss = -cos_similarity_cubed_single(conceptdis, dot_product)
        # dis_loss = th.mean(dis_loss)
        # min-max normalize
        
        normalized_conceptdis = (conceptdis - th.min(conceptdis)) / (th.max(conceptdis) - th.min(conceptdis))
        normalized_dot_product = (dot_product - th.min(dot_product)) / (th.max(dot_product) - th.min(dot_product))
        dis_loss = self.kl_divergence(normalized_conceptdis, normalized_dot_product)
        
        #emd_loss = self.wasserstein_distance(normalized_conceptdis, normalized_dot_product).mean()
 
        
        #concept_uncertainty = concept_uncertainty.mean()
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        #val_loss = loss + dis_loss
        val_loss = loss + self.cfg.lamba*dis_loss
        self.log('val_cls_loss', loss)
        self.log('val_dis_loss',dis_loss)
        self.log('val_all_loss',val_loss)
        #self.log('emd_loss',emd_loss)
        #self.log('val_loss',val_loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.valid_acc(pred, y)
        self.log('val_acc', self.valid_acc, on_step=False, on_epoch=True)
        th.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        #sim = self.forward(image)
        sim, conceptdis, dot_product, concept_uncertainty = self.forward(image)
        pred = 100 * sim
        loss = F.cross_entropy(pred, y)
        y_pred = pred.argmax(dim=-1)
        self.confmat(y_pred, y)
        self.all_y.append(y)
        self.all_pred.append(y_pred)
        self.log('test_loss', loss)
        self.log('concept_uncertainty',concept_uncertainty)
        self.test_acc(pred, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        return loss

    def kl_divergence(self, p, q):
        epsilon = 1e-9
        p = p + epsilon
        q = q + epsilon
        
        kl_elements_pq = p * (th.log(p) - th.log(q))
        kl_elements_qp = q * (th.log(q) - th.log(p))
        
        kl_elements_pq = th.where(th.isfinite(kl_elements_pq), kl_elements_pq, th.tensor(0.0, device=kl_elements_pq.device))
        kl_elements_qp = th.where(th.isfinite(kl_elements_qp), kl_elements_qp, th.tensor(0.0, device=kl_elements_qp.device))
        
        kl_pq = kl_elements_pq.sum(dim=1)  # 沿着特征维度求和
        kl_qp = kl_elements_qp.sum(dim=1)  # 沿着特征维度求和
        
        kl_loss = kl_pq + kl_qp
        return kl_loss.mean()  # 取平均值
        
        # kl_pq = F.kl_div(q.log(), p, reduction='sum')
        # kl_qp = F.kl_div(p.log(), q, reduction='sum')
        # kl_loss = kl_pq + kl_qp
        # return kl_loss.mean()
    
    def wasserstein_distance(self, p, q):
        # 计算累积分布函数
        cdf_p = th.cumsum(p, dim=1)
        cdf_q = th.cumsum(q, dim=1)

        # 计算Wasserstein距离
        wasserstein_distance = th.sum(th.abs(cdf_p - cdf_q), dim=1)
        return wasserstein_distance


    def customContraLoss(self, y_pred, y_true, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos = (y_true * log_prob).sum(1) / (y_true.sum(1) + eps)
        # Print intermediate values for debugging
        # print("logits:", logits)
        # print("exp_logits:", exp_logits)
        # print("log_prob:", log_prob)
        # print("mean_log_prob_pos:", mean_log_prob_pos)
        return -mean_log_prob_pos.mean()

    def customContraLoss2(self, y_pred, y_true1,y_true2, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
         
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()
  
    def customContraLoss3(self, y_pred, y_true1, y_true2,  y_true3, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()
    
    def customContraLoss4(self, y_pred, y_true1, y_true2,  y_true3, y_true4, tau, eps=1e-6):
        logits = y_pred / tau + eps
        logits = logits - th.max(logits, dim=1, keepdim=True)[0]
        exp_logits = th.exp(logits)
        log_prob = logits - th.log(exp_logits.sum(1, keepdim=True) + eps)
        mean_log_prob_pos1 = (y_true1 * log_prob).sum(1) / (y_true1.sum(1) + eps)
        mean_log_prob_pos2 = (y_true2 * log_prob).sum(1) / (y_true2.sum(1) + eps)
        mean_log_prob_pos3 = (y_true3 * log_prob).sum(1) / (y_true3.sum(1) + eps)
        mean_log_prob_pos4 = (y_true4 * log_prob).sum(1) / (y_true4.sum(1) + eps)
        return -mean_log_prob_pos1.mean()-mean_log_prob_pos2.mean()-mean_log_prob_pos3.mean()-mean_log_prob_pos4.mean()

