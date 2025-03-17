import torch
from torch import nn
class SceneModule(nn.Module):
    def forward(self, scene_features):
        """
        scene_features: Tensor of shape (N_obj, D_obj) from PointTokenizeEncoder.
        Returns all object features (possibly with spatial encodings). 
        """
        # In simplest form, just return the features as the set of all objects.
        return scene_features  # shape: (N_obj, D_obj)

class FilterModule(nn.Module):
    def __init__(self, d_obj, d_txt):
        super().__init__()
        # Project text embedding into object feature space for similarity comparison
        self.text_proj = nn.Linear(d_txt, d_obj)
        # Optionally, a small MLP or attention layer can be used for more complex fusion
        # self.fusion = UnifiedSpatialCrossEncoderV2(d_obj, d_txt, ...)  # pseudo-code
    
    def forward(self, object_feats, text_feat):
        """
        object_feats: Tensor (N_obj, D_obj) for objects to filter (could be from scene() or previous module).
        text_feat: Tensor (D_txt) for the filter query (e.g. BERT encoding of attribute or category phrase).
        """
        # Project text query into object feature dimension
        query_vec = self.text_proj(text_feat)               # shape: (D_obj)
        # Compute similarity scores between query and each object
        scores = (object_feats * query_vec).sum(dim=1)      # shape: (N_obj,), dot product similarity
        weights = torch.softmax(scores, dim=0)              # shape: (N_obj,), attention weights over objects
        # Produce filtered set: we keep object features weighted by similarity (soft selection)
        filtered_feats = weights.unsqueeze(1) * object_feats  # shape: (N_obj, D_obj)
        return filtered_feats, weights

class RelateModule(nn.Module):
    def __init__(self, d_obj, d_txt):
        super().__init__()
        # Learnable fusion layers for combining anchor, relation text, and candidate features
        self.anchor_proj = nn.Linear(d_obj, d_obj)   # to transform anchor object feature
        self.rel_text_proj = nn.Linear(d_txt, d_obj) # to transform relation text embedding
        self.score_mlp = nn.Sequential(              # to compute compatibility score for each candidate
            nn.Linear(2*d_obj + 3, 128),  # 3 extra dims for spatial offset (dx, dy, dz for example)
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, all_objects, anchor_feats, relation_text, coords):
        """
        all_objects: Tensor (N_obj, D_obj) for all scene objects (potential targets).
        anchor_feats: Tensor (M, D_obj) for anchor object(s) (from OBJs input).
        relation_text: Tensor (D_txt) embedding of relation query (e.g., "next to", "on top of").
        coords: Tensor (N_obj, 3) of (x,y,z) coordinates for each object (and similarly we assume anchor(s) have coordinates).
        """
        # Aggregate anchor features into one representation (e.g., mean if multiple anchors)
        anchor_rep = anchor_feats.mean(dim=0)                     # shape: (D_obj)
        # Project anchor and relation text into common space
        anchor_vec = self.anchor_proj(anchor_rep)                 # shape: (D_obj)
        rel_vec = self.rel_text_proj(relation_text)               # shape: (D_obj)
        # Combine anchor and relation into a single query vector
        query_vec = torch.tanh(anchor_vec + rel_vec)              # shape: (D_obj)
        # Compute a score for each candidate object relative to the anchor
        # We concatenate the query, each object's features, and spatial offset (anchor->object)
        anchor_coord = (anchor_feats[0] if anchor_feats.dim()>1 else anchor_feats).new_tensor(coords[anchor_feats.argmax(dim=0)])  
        # (For simplicity, assume anchor_feats corresponds to a single anchor here and anchor_coord is its coordinate.)
        spatial_offset = coords - anchor_coord                    # shape: (N_obj, 3)
        # Form input for score MLP for each object
        num_objs = all_objects.shape[0]
        query_expand = query_vec.unsqueeze(0).expand(num_objs, -1)          # (N_obj, D_obj)
        combined_feats = torch.cat([query_expand, all_objects, spatial_offset], dim=1)  # (N_obj, 2*D_obj+3)
        scores = self.score_mlp(combined_feats).squeeze(-1)       # shape: (N_obj,), raw relation scores
        weights = torch.softmax(scores, dim=0)                    # attention over objects
        related_feats = weights.unsqueeze(1) * all_objects        # weighted object features of targets
        return related_feats, weights

def AndModule(mask_A, mask_B):
    # mask_A, mask_B: Tensor (N_obj,) with values in [0,1] indicating object selection probabilities.
    return mask_A * mask_B  # Intersection mask

def OrModule(mask_A, mask_B):
    # Union mask (soft)
    return torch.clamp(mask_A + mask_B, max=1.0)

class UnifiedSpatialCrossEncoderV2(nn.Module):
    def __init__(self, d_obj, d_txt, num_layers=1):
        super().__init__()
        # one transformer encoder layer or decoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_obj, nhead=4)  # object self-attn (optional)
        self.obj_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_obj, nhead=4)
        self.cross_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # project text embedding to d_obj if needed
        self.text_proj = nn.Linear(d_txt, d_obj)
    def forward(self, object_feats, text_feat):
        # Optionally refine object feats (could incorporate self-attention among objects)
        obj_memory = self.obj_encoder(object_feats.unsqueeze(1)).squeeze(1)  # shape: (N_obj, d_obj)
        # Prepare text query as sequence (here just one token)
        query = self.text_proj(text_feat).unsqueeze(0).unsqueeze(1)  # shape: (1, 1, d_obj)
        # Cross-attend text query to object memory
        out = self.cross_decoder(query, obj_memory.unsqueeze(1))    # shape: (1, 1, d_obj)
        attn_weights = self.cross_decoder.layers[-1].multihead_attn_weights  # last layer attn weights
        # attn_weights might be shape (1, N_obj) for the query token over objects
        return out.squeeze(1), attn_weights.squeeze(0)

class AnswerDecoder(nn.Module):
    def __init__(self, d_model, num_layers, vocab_size):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=8, dim_feedforward=2048)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.token_embed = nn.Embedding(vocab_size, d_model)  # embed target tokens (answers)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, encoder_memory, target_seq):
        """
        encoder_memory: Tensor (L_mem, 1, D_model) from encoder/fusion modules (memory that decoder attends to).
                        This could include question context and visual features.
        target_seq: Tensor (L_ans, 1) of token indices for training (teacher-forcing). 
                    For inference, we would generate this step by step.
        """
        # Embed the target (previous answer tokens or <SOS> start)
        tgt_emb = self.token_embed(target_seq)  # (L_ans, 1, D_model)
        # Decoder cross-attends to the encoder_memory (which encodes question + scene info)
        decoder_out = self.decoder(tgt_emb, encoder_memory)  # (L_ans, 1, D_model)
        logits = self.output_proj(decoder_out)  # (L_ans, 1, vocab_size)
        return logits
