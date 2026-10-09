PATCHES = [("""        if val_acc > best_acc and val_loss < best_loss:
            best_acc = val_acc
            best_loss = val_loss
            torch.save(model.state_dict(), best_state_path)
            epochs_without_improve = 0
""", """        if val_acc >= best_acc or float(val_loss) <= best_loss:
            if val_acc >= best_acc and float(val_loss) <= best_loss:
                torch.save(model.state_dict(), best_state_path)
            best_acc = max(best_acc, val_acc); best_loss = min(best_loss, float(val_loss))
            epochs_without_improve = 0
""")]
