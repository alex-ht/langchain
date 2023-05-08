from transformers import  FeatureExtractionPipeline

class SgptPipeline(FeatureExtractionPipeline):
    def _forward(self, model_inputs):
        import torch
        with torch.no_grad():
        # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = self.model(**model_inputs, output_hidden_states=True, return_dict=True).last_hidden_state
        
        # Get weights of shape [bs, seq_len, hid_dim]
        
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            model_inputs["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask
        return embeddings
    
    def postprocess(self, model_outputs, return_tensors=False):
        if return_tensors:
            return model_outputs
        if self.framework == "pt":
            return model_outputs.tolist()
        elif self.framework == "tf":
            return model_outputs.numpy().tolist()