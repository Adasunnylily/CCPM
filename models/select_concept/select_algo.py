import torch as th
import random
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import OrderedDict

def get_tSNE_embed(x):
    tSNE = TSNE(n_components=2, init='random')
    return tSNE.fit_transform(x)




### 越小越好
def cal_enc_avg_cls_score(scores_mean):
    enc_avg_cls_score = th.var(scores_mean, dim=0)
    return enc_avg_cls_score

def cal_inter_class_score(scores_mean):
    inter_class_score = th.var(scores_mean, dim=1)
    return inter_class_score

def cal_intra_class_score(img_feat, concept_feat, n_shots, num_images_per_class):
    num_cls = len(num_images_per_class)
    intra_class_score = th.empty((concept_feat.shape[0],num_cls))
    start_loc = 0
    #img_feat = img_feat[:,0,:]
    for i in range(num_cls):
        end_loc = sum(num_images_per_class[:i+1])
        #intra_class_score[:, i] = (concept_feat @ img_feat[:,0,:][start_loc:end_loc].t()).var(dim=-1)
        intra_class_score[:, i] = (concept_feat @ img_feat[start_loc:end_loc].t()).var(dim=-1)
        start_loc = end_loc
    return intra_class_score.mean(dim=-1)

def cal_all_image_var_score(img_feat, concept_feat, n_shots, num_images_per_class):
    #intra_class_score = th.empty((concept_feat.shape[0]))
    #all_image_var_score = concept_feat @ img_feat[:,0,:].t().var(dim=-1)
    all_image_var_score = concept_feat @ img_feat.t().var(dim=-1)
    return all_image_var_score









def clip_score(img_feat, concept_feat, n_shots, num_images_per_class):
    num_cls = len(num_images_per_class)
    scores_mean = th.empty((concept_feat.shape[0], num_cls))
    start_loc = 0
    for i in range(num_cls):
        end_loc = sum(num_images_per_class[:i+1])
        end_loc = int(end_loc)
        #scores_mean[:, i] = (concept_feat @ img_feat[:,0,:][start_loc:end_loc].t()).mean(dim=-1)
        scores_mean[:, i] = (concept_feat @ img_feat[start_loc:end_loc].t()).mean(dim=-1)
        start_loc = end_loc
    return scores_mean


def mi_score(img_feat, concept_feat, n_shots, num_images_per_class):
    num_cls = len(num_images_per_class)
    scores_mean = clip_score(img_feat, concept_feat, n_shots, num_images_per_class) # Sim(c,y)
    normalized_scores = scores_mean / (scores_mean.sum(dim=0) * num_cls) # Sim_bar(c,y)
    margin_x = normalized_scores.sum(dim=1) # sum_y in Y Sim_bar(c,y)
    margin_x = margin_x.reshape(-1, 1).repeat(1, num_cls)
    # compute MI and PMI
    pmi = th.log(normalized_scores / (margin_x * 1 / num_cls)) # log Sim_bar(c,y) / sum_y in Y Sim_bar(c,y) / N = log(Sim_bar(c|y))
    mi = normalized_scores * pmi  # Sim_bar(c,y)* log(Sim_bar(c|y))
    mi = mi.sum(dim=1)
    return mi, scores_mean


def mi_select(img_feat, concept_feat, n_shots, num_images_per_class, *args):
    mi, _ = mi_score(img_feat, concept_feat, n_shots, num_images_per_class)
    _, selected_idx = th.sort(mi, descending=True)
    return selected_idx


def clip_score_select(img_feat, concept_feat, n_shots, num_images_per_class, *args):
    scores_mean = clip_score(img_feat, concept_feat, n_shots, num_images_per_class)
    best_scores_over_cls = scores_mean.max(dim=-1)[0]
    _, selected_idx = th.sort(best_scores_over_cls, descending=True)
    return selected_idx


def group_clip_select(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, *args):
    assert num_concepts > 0
    num_cls = len(num_images_per_class)
    scores = clip_score(img_feat, concept_feat, n_shots, num_images_per_class).max(dim=-1)[0]

    selected_idx = []
    concept2cls = th.from_numpy(concept2cls).long()
    num_concepts_per_cls = num_concepts // num_cls
    for i in tqdm(range(num_cls)):
        cls_idx = th.where(concept2cls == i)[0]
        _, idx_for_cls_idx = th.topk(scores[cls_idx], num_concepts_per_cls)
        global_idx = cls_idx[idx_for_cls_idx]
        selected_idx.extend(global_idx)
    return th.tensor(selected_idx)


def group_mi_select(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    assert num_concepts > 0
    num_cls = len(num_images_per_class)
    scores, _ = mi_score(img_feat, concept_feat, n_shots, num_images_per_class)
    take_all = False
    selected_idx = []
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    concept2cls = th.from_numpy(concept2cls).long()
    for i in tqdm(range(num_cls)):
        cls_idx = th.where(concept2cls == i)[0]
        if len(cls_idx) == 0: continue

        elif len(cls_idx) < num_concepts_per_cls or (take_all and num_cls < 10):
            global_idx = cls_idx

        else:
            _, idx_for_cls_idx = th.topk(scores[cls_idx], num_concepts_per_cls)
            global_idx = cls_idx[idx_for_cls_idx]

        selected_idx.extend(global_idx)
    return th.tensor(selected_idx)


def clip_score_select_within_cls(img_feat, concept_feat, n_shots, concept2cls):
    # taking from cls2concept and then select top concept within each class
    num_cls = len(img_feat) // n_shots
    scores = concept_feat @ img_feat.t()
    scores = scores.view(concept_feat.shape[0], num_cls, n_shots)
    scores_mean = scores.mean(dim=-1) # (num_concept, num_cls)
    init_cls_id = list(concept2cls.values()) # (num_concept, 1)
    init_cls_id = th.tensor(init_cls_id).view(-1, 1)
    init_score = th.gather(scores_mean, 1, init_cls_id)
    _, selected_idx = th.sort(init_score, descending=True)
    return selected_idx


def compute_class_similarity(img_feat, n_shots):
    # img_feat: n_shots * num_cls x d
    # img_sim: n_shots * num_cls x n_shots * num_cls
    num_cls = len(img_feat) // n_shots
    img_sim = img_feat @ img_feat.T
    class_sim = th.empty((num_cls, num_cls), dtype=th.long)
    for i, row_split in enumerate(th.split(img_sim, n_shots, dim=0)):
        for j, col_split in enumerate(th.split(row_split, n_shots, dim=1)):
            class_sim[i, j] = th.mean(col_split)
    return class_sim / class_sim.max(dim=0).values


def plot(features, selected_idx, filename):
    tsne_features = get_tSNE_embed(features)
    x_selected = tsne_features[selected_idx,0]
    y_selected = tsne_features[selected_idx,1]
    x = tsne_features[:,0]
    y = tsne_features[:,1]
    plt.clf()
    plt.scatter(x, y, s = 1, c ="blue")
    plt.scatter(x_selected, y_selected, s = 10, c ="red", alpha=1)
    plt.savefig('{}.png'.format(filename))









import tracemalloc
import time
 
def RAS_wcls(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    from apricot import CustomSelection, MixtureSelection, FacilityLocationSelection
    assert num_concepts > 0
    num_cls = len(num_images_per_class)
    
    
    # 开始跟踪内存使用
    tracemalloc.start()
    start_time = time.time()
    
    # 计算浮点运算次数
    flops = 0
    
    
    all_indices = set(range(concept_feat.shape[0]))
    ### all_mi_score shape: number of concepts
    #all_mi_scores, _ = mi_score(img_feat, concept_feat, n_shots, num_images_per_class)
    selected_idx = []
    num_concepts_per_cls = int(np.floor(num_concepts / num_cls))

    # def mi_based_function(X):
    #     return X[:, 0].sum()
    scores_mean = clip_score(img_feat, concept_feat, n_shots, num_images_per_class)
    #flops += scores_mean.numel()
    inter_class_score = cal_inter_class_score(scores_mean)
    intra_class_score = cal_intra_class_score(img_feat, concept_feat, n_shots, num_images_per_class)
    all_image_var_score = cal_all_image_var_score(img_feat, concept_feat, n_shots, num_images_per_class)
   
    # 累加计算上面函数调用的 FLOPs
    flops += scores_mean.numel() + inter_class_score.numel() + intra_class_score.numel() + all_image_var_score.numel()
     
   
    a = 1
    b = 1
    c = 1
    all_var_score = a*inter_class_score - b*intra_class_score + c*all_image_var_score

    flops += all_var_score.numel()

    def var_based_function(X):
        nonlocal flops
        flops += X.shape[0] * X.shape[1]  # 假设每个元素进行一次浮点运算
        #flops += X.shape[0] * X.shape[1]  # 假设每个元素进行一次浮点运算
        return X[:, 0].sum()
    
    #mi_selector = CustomSelection(num_concepts_per_cls, mi_based_function)
    distance_selector = FacilityLocationSelection(num_concepts_per_cls, metric='cosine')
    variance_selector = CustomSelection(num_concepts_per_cls, var_based_function)

    facility_weight = submodular_weights[2]
    variance_weight = submodular_weights[1]
    var_scores = all_var_score * variance_weight
    
    # if mi_score_scale == 0:
    #     submodular_weights = [0, facility_weight]
    # else:
    submodular_weights = [variance_weight, facility_weight]
     
 
    #concept2cls = th.from_numpy(concept2cls).long()
    # 初始化已选择的索引集合
    selected_idx_set = set()


    if concept2cls.ndim == 2:
        print("concept2cls 的维度是2")
        concept2cls = concept2cls.t().long()
        
        for i in tqdm(range(num_cls)):
            #cls_idx = th.where(concept2cls == i)[0]
            cls_idx = th.nonzero(concept2cls[i,:], as_tuple=False)
            # 检查是否有重复索引
            cls_idx = [idx.item() for idx in cls_idx if idx.item() not in selected_idx_set]

            if len(cls_idx) <= num_concepts_per_cls:
                selected_idx.extend(cls_idx)
                selected_idx_set.update(cls_idx)
            else:
                #mi_scores = all_mi_scores[cls_idx] * mi_score_scale

                current_concept_features = concept_feat[cls_idx,:]
                current_var_scores = var_scores[cls_idx]
                
                augmented_concept_features = th.hstack([th.unsqueeze(current_var_scores, 1), current_concept_features]).numpy()
                selector = MixtureSelection(num_concepts_per_cls, functions=[variance_selector, distance_selector], weights=submodular_weights, optimizer='naive', verbose=False)
                
                selected = selector.fit(augmented_concept_features).ranking
                selected_idx.extend(np.array(cls_idx)[selected])
                selected_idx_set.update(np.array(cls_idx)[selected])
        
    
    else:
        concept2cls = th.from_numpy(concept2cls).long()
      
    
        print(concept2cls.shape)
        #concept2cls = concept2cls.t().long()
    

        for i in tqdm(range(num_cls)):
            #cls_idx = th.where(concept2cls == i)[0]
            #cls_idx = th.nonzero(concept2cls[i,:], as_tuple=False)
            #cls_idx = th.nonzero(concept2cls, as_tuple=False)
            cls_idx = th.where(concept2cls == i)
            
            
            # 检查是否有重复索引
            cls_idx = [idx.item() for idx in cls_idx[0] if idx.item() not in selected_idx_set]

            if len(cls_idx) <= num_concepts_per_cls:
                selected_idx.extend(cls_idx)
                selected_idx_set.update(cls_idx)
            else:
                #mi_scores = all_mi_scores[cls_idx] * mi_score_scale

                current_concept_features = concept_feat[cls_idx,:]
                current_var_scores = var_scores[cls_idx]
                
                augmented_concept_features = th.hstack([th.unsqueeze(current_var_scores, 1), current_concept_features]).numpy()
                selector = MixtureSelection(num_concepts_per_cls, functions=[variance_selector, distance_selector], weights=submodular_weights, optimizer='naive', verbose=False)
                
                selected = selector.fit(augmented_concept_features).ranking
                selected_idx.extend(np.array(cls_idx)[selected])
                selected_idx_set.update(np.array(cls_idx)[selected])
        
        
    already_all_select = False
        
    print('already select',len(selected_idx_set))
   
    if len(selected_idx_set) < num_concepts:
        num_to_select = num_concepts - len(selected_idx_set)
        
        print('need to select',num_to_select)
        
        unselected_index = list(all_indices - selected_idx_set)
        
        if len(unselected_index) < num_to_select:
            selected_idx_set.update(np.array(unselected_index))
            already_all_select = True
        else:
            variance_selector = CustomSelection(num_to_select, var_based_function)
            distance_selector = FacilityLocationSelection(num_to_select, metric='cosine')

            #mi_score_scale = submodular_weights[0]
            # facility_weight = submodular_weights[2]
            # variance_weight = submodular_weights[1]

            # Scale MI scores
            #mi_scores = all_mi_scores * mi_score_scale
            var_scores = all_var_score * variance_weight

            current_concept_features = concept_feat[unselected_index]
            var_scores = var_scores[unselected_index]
            augmented_concept_features = th.hstack([th.unsqueeze(var_scores, 1),current_concept_features]).numpy()

            # Use MixtureSelection for submodular optimization
            selector = MixtureSelection(
                num_to_select,
                functions=[variance_selector, distance_selector],
                weights=[1, facility_weight],
                optimizer='naive',
                verbose=False
            )
            selected = selector.fit(augmented_concept_features).ranking
            print('new_casual_selected',selected.shape)
            selected_idx_set.update(np.array(unselected_index)[selected])
            #selected_idx.extend(selected)
    
    print("列表中达到",len(selected_idx_set))

    # 结束内存使用跟踪
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    print("FLOPs:", flops)
    print(f"Current memory usage: {current / 10**6}MB; Peak: {peak / 10**6}MB")
    print(f"Execution time: {end_time - start_time} seconds")
    


    if not already_all_select:
        
        assert len(selected_idx_set) == num_concepts

    if len(selected_idx_set) == 400:
        print("列表中达到400")
     
    # 将集合转换为有序字典的键
    selected_idx_ordered_dict = OrderedDict.fromkeys(selected_idx_set)
    
    

    # 将有序字典的键转换为列表
    selected_idx_list = list(selected_idx_ordered_dict.keys())
    
    print('实际选取的数目',len(selected_idx_list))
    return th.tensor(selected_idx_list)


 
def RAS_wocls(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    from apricot import CustomSelection, MixtureSelection, FacilityLocationSelection, SumRedundancySelection
    assert num_concepts > 0
    num_cls = len(num_images_per_class)

    # # Calculate mutual information (MI) scores
    # all_mi_scores, _ = mi_score(img_feat, concept_feat, n_shots, num_images_per_class)
    selected_idx = []

    # def mi_based_function(X):
    #     return X[:, 0].sum()

    def var_based_function(X):
        return X[:, 0].sum()
    
    if img_feat.dtype == th.float16:
        img_feat = img_feat.to(th.float32).detach()

    
    scores_mean = clip_score(img_feat, concept_feat, n_shots, num_images_per_class)
    inter_class_score = cal_inter_class_score(scores_mean)
    intra_class_score = cal_intra_class_score(img_feat, concept_feat, n_shots, num_images_per_class)
    all_image_var_score = cal_all_image_var_score(img_feat, concept_feat, n_shots, num_images_per_class)
   
    a = 1
    b = 1
    c = 1
    all_var_score = a*inter_class_score - b*intra_class_score + c*all_image_var_score



    #mi_selector = CustomSelection(num_concepts, mi_based_function)
    distance_selector = FacilityLocationSelection(num_concepts, metric='cosine')
    #variance_selector = SumRedundancySelection(num_concepts, metric='euclidean')
    variance_selector = CustomSelection(num_concepts, var_based_function)

    #mi_score_scale = submodular_weights[0]
    facility_weight = submodular_weights[2]
    variance_weight = submodular_weights[1]

    # Scale MI scores
    #mi_scores = all_mi_scores * mi_score_scale
    var_scores = all_var_score * variance_weight

    #concept2cls = th.from_numpy(concept2cls).long()

    # # Ensure selecting a fixed number of concepts across all selected classes
    # num_concepts_total = min(num_concepts, num_cls * len(set(concept2cls)))
    
    current_concept_features = concept_feat
    augmented_concept_features = th.hstack([th.unsqueeze(var_scores, 1),current_concept_features]).numpy()

    # Use MixtureSelection for submodular optimization
    selector = MixtureSelection(
        num_concepts,
        functions=[variance_selector, distance_selector],
        weights=[1, facility_weight],
        optimizer='naive',
        verbose=False
    )

    selected = selector.fit(augmented_concept_features).ranking
    selected_idx.extend(selected)

     

    if len(selected_idx) == len(set(selected_idx)):
        print("列表中没有重复项")
    else:
        print("列表中有重复项")
    print('已经选择',len(selected_idx))

    return th.tensor(selected_idx)


  
def random_select(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    selected_idxes = np.random.choice(np.arange(len(concept_feat)), size=num_concepts, replace=False)
    return th.tensor(selected_idxes)

def lm4cv(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    #selected_idxes = np.random.choice(np.arange(len(concept_feat)), size=num_concepts, replace=False)
    print('我们要选择concept数目',num_concepts)
    selected_idxes = th.load('lm4cv_204/select_idx.pth')
 
    print(selected_idxes.shape)
    return  selected_idxes


# import open_clip
from sklearn.cluster import KMeans
#only depend on the concept_embedding features
def kmeans_select(img_feat, concept_feat, n_shots, concept2cls, num_concepts, num_images_per_class, submodular_weights, *args):
    kmeans = KMeans(n_clusters=num_concepts, random_state=0).fit(concept_feat)
    centers = kmeans.cluster_centers_

    selected_idxes = []
    for center in centers:
        center = center / th.tensor(center).norm().numpy()
        distances = np.sum((concept_feat.numpy() - center.reshape(1, -1)) ** 2, axis=1)
        # sorted_idxes = np.argsort(distances)[::-1]
        sorted_idxes = np.argsort(distances)
        count = 0
        while sorted_idxes[count] in selected_idxes:
            count += 1
        selected_idxes.append(sorted_idxes[count])
    selected_idxes = np.array(selected_idxes)
    return th.tensor(selected_idxes)