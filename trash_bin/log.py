from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

accelerator = Accelerator(log_with='tensorboard', project_dir='./log')
accelerator.init_trackers('test')
#writer = SummaryWriter('./log')

for i in range(10):
    value1 = i**2
    value2 = 2*i
    accelerator.log({'i**2': value1}, step=i)
    #writer.add_scalar('try', value1, global_step=i)

#writer.close()
accelerator.end_training()