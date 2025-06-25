import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torchvision
torchvision.disable_beta_transforms_warning()
from typing import Optional, Literal, Union
from safetensors.torch import save_file
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class LoRALayerBase:
    
    def __init__(self,
                 rank=8,
                 lora_alpha=8,
                 lora_dropout=0.0,
                 use_rslora=True):
        
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x
        self.use_rslora = use_rslora
        
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank
        
    def _load_pretrained_weights(self, state_dict):  ##copy the weights
        self.weight.data = state_dict["weight"]
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bias"]
            

class LoRALinear(nn.Linear, LoRALayerBase):
    
    def __init__(self,
                in_features,
                out_features,
                bias = True,
                rank= 8,
                lora_alpha =8,
                lora_dropout=0.0,
                use_rslora=True,
                **kwargs):
        
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)    
        # print(self.weight)
        # print(self.bias)
        LoRALayerBase.__init__(self,
                               rank=rank,
                               lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout,
                               use_rslora=use_rslora)
        
        self.weight.requires_grad = False
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
    
    def _merge_weights(self):#returning w prime
        
        merged_weights =  self.weight.data + self.scaling * (self.lora_A @ self.lora_B).T
        state_dict = {"weight" : merged_weights}
        if self.bias is not None:
            state_dict["bias"] =  self.bias
            
        merged_linear = nn.Linear(self.in_features,
                                  self.out_features,
                                  bias=True if self.bias is not None else False)
        
        merged_linear.load_state_dict(state_dict)
        
        return merged_linear
        
    def forward(self,x):
        orig_layer_out = F.linear(x, self.weight, bias = self.bias)
        
        lora_mult = (self.lora_A @ self.lora_B) *self.scaling
        # print(lora_mult.shape)
        # print(self.weight.shape) #flipped size coz in pytorch they flip the matrix before multiplication
        low_rank_out = self.lora_dropout(x) @ lora_mult
        # print(low_rank_out.shape)
        # print(orig_layer_out.shape)
        output = orig_layer_out + low_rank_out
        return output  
    
   
class LoRAEmbedding(nn.Embedding, LoRALayerBase):
    
    """
    LoRA Implementation on an Embedding Layer, basically a wrapper on the
    nn.Embedding module, with the additional lora_A/lora_B layers
    """

    def __init__(self, 
                 num_embeddings, 
                 embedding_dim, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
        ### Initialize Inherited Classes ###
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayerBase.__init__(self,
                               rank=rank, 
                               lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, 
                               use_rslora=use_rslora)
        
        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        ### Disable Gradients on Linear Layer Weights ###
        self.weight.requires_grad = False
        # print(self.weight.shape)
        # Define LoRA Layers ###
        self.lora_A = nn.Parameter(torch.zeros(self.num_embeddings, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.embedding_dim))
        # print(self.lora_A.shape)
        # print(self.lora_B.shape)
        ### Initialize lora_A. In the paper they use normal, but in the ###
        ### implementation they use kaiming_uniform_, so lets use that! ###
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):
        """
        xW^T + xAB = x(W^T + AB)
        """

        ### Merge Weights ###
        merged_weights = self.weight.data + (self.lora_A @ self.lora_B) * self.scaling

        ### Store in Embedding Layer ###
        state_dict = {"weight": merged_weights}
        merged_emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        merged_emb.load_state_dict(state_dict)

        return merged_emb


    def forward(self, x):

        ### Pass Through Original Weights ###
        orig_layer_out = F.embedding(input=x, 
                                     weight=self.weight,
                                     padding_idx=self.padding_idx, 
                                     max_norm=self.max_norm, 
                                     norm_type=self.norm_type, 
                                     scale_grad_by_freq=self.scale_grad_by_freq, 
                                     sparse=self.sparse) #default extra stuff
        
        ### Pass Through lora_A ###
        low_rank_A_output = F.embedding(input=x, 
                                        weight=self.lora_A, 
                                        padding_idx=self.padding_idx, 
                                        max_norm=self.max_norm, 
                                        norm_type=self.norm_type, 
                                        scale_grad_by_freq=self.scale_grad_by_freq, 
                                        sparse=self.sparse)
        
        ### Project lora_A rank dimension to embedding dimension w/ lora_B ###
        low_rank_output = (low_rank_A_output @ self.lora_B) * self.scaling

        ### Sum Outputs ###
        output = orig_layer_out + low_rank_output

        return output


class LoRAConv2d(nn.Conv2d, LoRALayerBase):
    
    """
    LoRA Implementation on an Conv2d Layer, basically a wrapper on the
    nn.Conv2d module, with the additional lora_A/lora_B layers

    This is a simplified convolution implementation, take a look at the original
    implementation for more details!
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 bias=True,
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
        ### Initialize Inherited Classes ###
        nn.Conv2d.__init__(self,
                           in_channels=in_channels, 
                           out_channels=out_channels, 
                           kernel_size=kernel_size, 
                           stride=stride, 
                           padding=padding, 
                           bias=bias, 
                           **kwargs)
        LoRALayerBase.__init__(self,
                               rank=rank, 
                               lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, 
                               use_rslora=use_rslora)
        
        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        ### Disable Gradients on Linear Layer Weights/Biases ###
        self.weight.requires_grad = False
        # print(self.weight.shape)
        ### Create Our Low Rank Matricies (Flatten Kernel Weights and Output Rank) ###
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_channels, *self.kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_channels))
        # print(self.lora_A.shape)
        # print(self.lora_B.shape)
        ### Initialize lora_A. In the paper they use normal, but in the ###
        ### implementation they use kaiming_uniform_, so lets use that! ###
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):

        """
        xW^T + xAB = x(W^T + AB)
        """
        ### (rank x in_chan x k_h, k_w) -> (rank x in_chan*k_h*k_w) ###    
        lora_A_flatten = self.lora_A.flatten(1)

        ### Matmul with lora_B Transposed (lora_B is rank x out_channels) -> (out_channels x rank) ###
        lora_mult = (self.lora_B.T @ lora_A_flatten) * self.scaling
        
        ### Place Back into Conv Weight Shape: (ou_chan x in_chan*k_h*k_w) -> (out_chan x in_chan x k_h x k_w) ###
        lora_mult = lora_mult.reshape(self.out_channels, self.in_channels, *self.kernel_size)

        ### Merge ###
        merged_weight = self.weight.data + lora_mult

        ### Store Weights ###
        state_dict = {"weight": merged_weight}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        ### Load Weights to New Linear Layer ###
        merged_conv = nn.Conv2d(self.in_channels, 
                                self.out_channels,
                                kernel_size=self.kernel_size, 
                                stride=self.stride, 
                                padding=self.padding,
                                bias=True if self.bias is not None else False)
        
        merged_conv.load_state_dict(state_dict)

        return merged_conv

    def forward(self, x):

        ### Output W/ Original Weights ###
        orig_layer_out = F.conv2d(input=x, 
                                  weight=self.weight, 
                                  bias=self.bias,
                                  stride=self.stride, 
                                  padding=self.padding)
        print(orig_layer_out.shape)
        ### Low Rank Outputs ###
        lora_rank_A_output = F.conv2d(input=x, 
                                      weight=self.lora_A, 
                                      bias=None,
                                      stride=self.stride, 
                                      padding=self.padding)

        ### Permute to have rank_channels last (B x H x W x rank) ###
        lora_rank_A_output = lora_rank_A_output.permute(0,2,3,1)
        
        ### Multiply by lora_B (B x H x W x out_channels) ###
        low_rank_output = (self.lora_dropout(lora_rank_A_output) @ self.lora_B) * self.scaling

        ### Return Back to Image Shape (B x out_channels x H x W) ###
        low_rank_output = low_rank_output.permute(0,3,1,2)

        ### Sum Outputs ###
        output = orig_layer_out + low_rank_output

        return output
        

@dataclass

@dataclass
class LoraConfig:
    target_modules: list = ("q_proj", "k_proj", "v_proj", "fc1", "fc2")  # Common CLIP transformer MLPs
    exclude_modules: list = ()  # Optional
    rank: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    use_rslora: bool = True
    bias: str = "none"  # Options: "none", "all", "lora_only"

class LoraModel(nn.Module):

    def __init__(self, model, config):
        super(LoraModel, self).__init__()

        self.lora_model = model
        self.config = config

        ### Ensure Taraget Modules/Exclude Modules are Lists ###
        if self.config.target_modules is None:
            self.config.target_modules = []
        elif isinstance(self.config.target_modules, str):
            self.config.target_modules = [self.config.target_modules]
        if self.config.exclude_modules is None:
            self.config.exclude_modules = []
        elif isinstance(self.config.exclude_modules, str):
            self.config.exclude_modules = [self.config.exclude_modules]

        ### Get Number of Trainable Parameters ###
        orig_trainable_params = self._compute_trainable_parameters()

        ### Disable All Grads in Model ###
        self._disable_all_grads()

        ### Apply LoRA to Target Modules ###
        self._apply_lora(self.lora_model)

        ### Toggle Bias Gradients ###
        self._toggle_bias_grad()

        ### Get LoRA Trainable Parameters ###
        lora_trainable_params = self._compute_trainable_parameters()

        print_string = ""
        print_string += f"Initial Parameters : {orig_trainable_params} || "
        print_string += f"LoRA Parameters : {lora_trainable_params} || "
        print_string += f"Trainable Proportion : {round(lora_trainable_params*100/orig_trainable_params, 2)}%"

        print(print_string)

    def forward(self, *inputs, **kwargs):

        """
        The forward function is the same, so a catchall here
        to pass all of our stuff from the forward methdod into
        our models forward method
        """

        return self.lora_model(*inputs, **kwargs)
    
    def _exclude_module_name_check(self, name):
        return any([ex in name for ex in self.config.exclude_modules])
    
    def _target_module_name_check(self, name):
        return any([tgt in name for tgt in self.config.target_modules])
    
    def _apply_lora(self, module):

        """
        Method to recursively replace all the layers in a model with LoraLayers
        """

        ### Recursively Go Through Model and Find Layers To Convert ###
        for name, child in module.named_children():
            
            ### Check if Layer is Included to Convert to LoRA ###
            if self._target_module_name_check(name):
                
                ### Convert Linear to LoRA ###
                if isinstance(child, nn.Linear):

                    new_layer = LoRALinear(in_features=child.in_features, 
                                           out_features=child.out_features, 
                                           bias=True if child.bias is not None else False,
                                           rank=self.config.rank,
                                           lora_alpha=self.config.lora_alpha, 
                                           lora_dropout=self.config.lora_dropout, 
                                           use_rslora=self.config.use_rslora)

                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Conv2d):

                    new_layer = LoRAConv2d(in_channels=child.in_channels, 
                                           out_channels=child.out_channels, 
                                           kernel_size=child.kernel_size, 
                                           stride=child.stride, 
                                           padding=child.padding, 
                                           bias=True if child.bias is not None else False,
                                           rank=self.config.rank, 
                                           lora_alpha=self.config.lora_alpha, 
                                           lora_dropout=self.config.lora_dropout, 
                                           use_rslora=self.config.use_rslora)
                    
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Embedding):

                    new_layer = LoRAEmbedding(num_embeddings=child.num_embeddings, 
                                              embedding_dim=child.embedding_dim, 
                                              rank=self.config.rank, 
                                              lora_alpha=self.config.lora_alpha, 
                                              lora_dropout=self.config.lora_dropout, 
                                              use_rslora=self.config.use_rslora)
                    
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

            ### If there are more children and its not an exclusion module, Recurse into them ###
            if (len(list(child.children())) > 0) and not any([ex in name for ex in self.config.exclude_modules]):
                self._apply_lora(child)

    def _toggle_bias_grad(self):

        """
        Method to turn off bias gradients depending on:
            - none:  Dont train any biases
            - all: train all biases
            - lora_only: train biases only in lora layers
        """

        for name, param in self.lora_model.named_parameters():
            
            ### Dont want to disable gradients for Excluded Layers ###
            if not self._exclude_module_name_check(name):
                if ".bias" in name:
                    if self.config.bias == "none":
                        param.requires_grad = False
                    elif self.config.bias == "all":
                        param.requires_grad = True
                    elif (self.config.bias == "lora_only") and self._target_module_name_check(name):
                        param.requires_grad = True

    def _disable_all_grads(self):
        
        """
        Helper function to disable all gradients 
        """

        for name, param in self.lora_model.named_parameters():

            ### If not in exclude modules, turn off gradients ###
            if not self._exclude_module_name_check(name):
                param.requires_grad = False

    def _compute_trainable_parameters(self):

        """
        Helper function to compute all parameters with gradients
        """

        total_learnable_params = 0
        for param in self.lora_model.parameters():
            if param.requires_grad:
                total_learnable_params += param.numel()

        return total_learnable_params

    def _merge_weights(self, module):

        """
        Recursively trigger weight merging and replace in model 
        """

        for name, child in module.named_children():

            if isinstance(child, (LoRALinear, LoRAEmbedding, LoRAConv2d)):
                 
                 ### Merge the Layer ###
                 merged_layer = child._merge_weights()

                 ### Replace LoRA Layer with Merged ###
                 setattr(module, name, merged_layer)

            else:

                if len(list(child.children())) > 0:
                    self._merge_weights(child)

    def save_model(self, path, merge_weights=False):

        """
        Method to save model safetensors to the given path
            - merge_weights -> True: Merge LoRA weights and save
            - merge_weights -> False: Only save trainable weights
        """

        def _detach_cpu(param):
            return param.detach().cpu()
        
        ### Create New Model with Merged Weights ###
        if merge_weights:
            
            ### Merge Weights ###
            self._merge_weights(self.lora_model)

            ### If Merged, then state_dict will have ALL Weights ###
            ### When merging weights, we can remove "lora_model." from the name ###
            ### because we can just load these weights into the original model ###
            state_dict = {name.replace("lora_model.", ""): _detach_cpu(param) for (name, param) in self.named_parameters()}

        ### Otherwise Save only the parameters we trained, everything else is frozen ###
        ### and can be taken from the original model weights ###
        ### To load these weights, the model needs to be wrapped in LoraModel ###
        else:

            state_dict = {name: _detach_cpu(param) for (name, param) in self.named_parameters() if (param.requires_grad)}

        save_file(state_dict, path)

if __name__ == "__main__":
    # layer = LoRALinear(4,4, rank =2)
    # layer = LoRAEmbedding(200,64,rank = 2)
    layer = LoRAConv2d(in_channels=256, out_channels=384,kernel_size=4)
    # rand = torch.randn(4,4)
    rand = torch.randn(4,256,64,64)
    layer(rand)
    # print(layer)
    # output = layer(rand)
    # print(output)
    
    # merged = layer._merge_weights()
    # output2 = merged(rand)
    # print(output2) #gives same resulter 
    from transformers import CLIPModel

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    print(clip)
    config = LoraConfig()
    lora_clip = LoraModel(clip, config)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, lora_clip.parameters()), lr=1e-4)
    print(lora_clip)
    
