import torch
import torch.nn as nn
import numpy as np

from sdf import SDF

class SDFLoss(nn.Module):

    def __init__(self, faces, grid_size=32, robustifier=None, debugging=False):
        super(SDFLoss, self).__init__()
        self.sdf = SDF()
        self.register_buffer('faces', torch.tensor(faces.astype(np.int32)))
        self.grid_size = grid_size
        self.robustifier = robustifier
        self.debugging = debugging

    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        num_people = vertices.shape[0]
        boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
        for i in range(num_people):
            boxes[i, 0, :] = vertices[i].min(dim=0)[0]
            boxes[i, 1, :] = vertices[i].max(dim=0)[0]
        return boxes

    @torch.no_grad()
    def check_overlap(self, bbox1, bbox2):
        # check x
        if bbox1[0,0] > bbox2[1,0] or bbox2[0,0] > bbox1[1,0]:
            return False
        #check y
        if bbox1[0,1] > bbox2[1,1] or bbox2[0,1] > bbox1[1,1]:
            return False
        #check z
        if bbox1[0,2] > bbox2[1,2] or bbox2[0,2] > bbox1[1,2]:
            return False
        return True

    def filter_isolated_boxes(self, boxes):

        num_people = boxes.shape[0]
        isolated = torch.zeros(num_people, device=boxes.device, dtype=torch.uint8)
        for i in range(num_people):
            isolated_i = False
            for j in range(num_people):
                if j != i:
                    isolated_i |= not self.check_overlap(boxes[i], boxes[j])
            isolated[i] = isolated_i
        return isolated

    def forward(self, vertices, translation, scale_factor=0.2):
        num_people = vertices.shape[0]
        # If only one person in the scene, return 0
        vertices = vertices + translation.unsqueeze(dim=1)
        loss = torch.tensor(0., device=vertices.device)
        if num_people == 1:
            return loss
        boxes = self.get_bounding_boxes(vertices)
        overlapping_boxes = ~self.filter_isolated_boxes(boxes)

        # If no overlapping voxels return 0
        if overlapping_boxes.sum() == 0:
            return loss

        # Filter out the isolated boxes
        vertices = vertices[overlapping_boxes, :].contiguous()
        translation = translation[overlapping_boxes, :].contiguous()
        boxes = boxes[overlapping_boxes]
        boxes_center = boxes.mean(dim=1).unsqueeze(dim=1)
        boxes_scale = (1+scale_factor) * 0.5*(boxes[:,1] - boxes[:,0]).max(dim=-1)[0][:,None,None]
        with torch.no_grad():
            vertices_centered = vertices - boxes_center
            vertices_centered_scaled = vertices_centered / boxes_scale
            assert(vertices_centered_scaled.min() >= -1)
            assert(vertices_centered_scaled.max() <= 1)
            phi = self.sdf(self.faces, vertices_centered_scaled)
            assert(phi.min() >= 0)

        valid_people = vertices.shape[0]
        # Convert vertices to the format expected by grid_sample
        for i in range(valid_people):
            weights = torch.ones(valid_people, 1, device=vertices.device)
            weights[i,0] = 0.
            # Change coordinate system to local coordinate system of each person
            vertices_local = (vertices - boxes_center[i].unsqueeze(dim=0)) / boxes_scale[i].unsqueeze(dim=0)
            vertices_grid = vertices_local.view(1,-1,1,1,3)
            # Sample from the phi grid
            phi_val = nn.functional.grid_sample(phi[i][None, None], vertices_grid).view(valid_people, -1)
            # ignore the phi values for the i-th shape
            cur_loss = weights * phi_val
            if self.debugging:
                import ipdb;ipdb.set_trace()
            # robustifier
            if self.robustifier:
                frac = (cur_loss / self.robustifier) ** 2
                cur_loss = frac / (frac + 1)

            loss += cur_loss.sum() / valid_people ** 2
        return loss
