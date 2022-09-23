import torch
import torch.nn as nn

from proj_name.models import MODELS
from proj_name.utils import load_args, prepare_datasets, prepare_dataloader


def train():

    args = load_args('configs.conf')

    usecuda = torch.cuda.is_available() and not args['no_cuda']
    data = prepare_datasets(**args, usecuda=usecuda)
    loaders = prepare_dataloader(data, **args)

    # Run wandb (logs training)
    if not args['no_wandb']:
        import wandb
        run = wandb.init(
            project='project-name',
            group="%s" % (args['dataset']),
            config=args,
            reinit=True
        )
    else:
        wandb = None

    device = torch.device("cuda") if usecuda else torch.device("cpu")
    print("Using device", device)

    assert args.model in MODELS, "Model doesn't exist"
    model = MODELS[args['model']](**args, device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    loss_fn = nn.L1Loss()

    # Training loop
    for epoch in range(args['epochs']):

        train_loss = 0

        for data in loaders['train']:

            data = data.to(device)
            optimizer.zero_grad()
            out = model(data, epoch)
            loss = loss_fn(out)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= (len(loaders['train']) * args['batch'])
    
        with torch.no_grad():

            model.eval()
            val_loss = 0

            for data in loaders['val']:

                data = data.to(device)
                out = model(data, epoch)
                loss = loss_fn(out)
                val_loss += loss.item()
            
            val_loss /= (len(loaders['val']) * args['batch'])
        
            model.train()
        
        # Log results in wandb
        if wandb is not None:
            wandb.log({"train-loss": train_loss, "val-loss": val_loss})

    # There's no test loop yet

    if wandb is not None:
        # Log final accuracy/test set results
        accuracy = 1
        wandb.run.summary['accuracy'] = accuracy
        run.finish()
    
    return [train_loss, val_loss, accuracy]



if __name__=='__main__':

    results = []
    for _ in range(10):
        result = train()
        results.append(result)