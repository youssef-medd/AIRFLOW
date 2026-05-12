import torch


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, save_path: str = "best.pt"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.save_path  = save_path
        self.best_score = None
        self.counter    = 0
        self.stop       = False

    def step(self, val_acc: float, model: torch.nn.Module) -> bool:
        score = val_acc
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def load_best(self, model: torch.nn.Module) -> torch.nn.Module:
        model.load_state_dict(torch.load(self.save_path, map_location="cpu"))
        return model
