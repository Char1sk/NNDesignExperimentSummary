import numpy as np
import torch
class MyQueue:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.__queue = [0 for _ in range(maxsize)]
        self.head = 0
        self.tail = 0
        self.__empty = True
        self.__full = False

    def __len__(self):
        if self.head <= self.tail:
            return self.tail - self.head
        else:
            return self.head + self.maxsize - self.tail - 1
    
    def full(self) -> bool:
        return self.__full

    def empty(self) -> bool:
        return self.__empty

    def push(self, item):
        if self.__full == True:
            self.head = (self.head + 1) % self.maxsize
        if (self.tail + 1) % self.maxsize == self.head:
            self.__full = True
        if self.__empty == True:
            self.__empty = False
        self.__queue[self.tail] = item
        self.tail = (self.tail + 1) % self.maxsize
        # print(">>>", self.__queue, " ", self.head, " ", self.tail, self.qsum())
    
    def pop(self):
        if self.__empty == True:
            raise ValueError("The queue is already EMPTY!")
        if (self.head + 1) % self.maxsize == self.tail:
            self.__empty = True
        if self.__full == True:
            self.__full = False
        item = self.__queue[self.head]
        self.head = (self.head + 1) % self.maxsize
        return item

    def qsum(self):
        sm = 0
        index = self.head
        flag = True
        while True:
            if index == self.tail:
                if flag:
                    flag = False
                else:
                    break
            sm += self.__queue[index]
            index = (index + 1) % self.maxsize
        return sm

    def qmin(self):
        min_value = 0x0fffffff
        index = self.head
        flag = True
        while True:
            if index == self.tail:
                if flag:
                    flag = False
                else:
                    break
            min_value = min(min_value, self.__queue[index])
            index = (index + 1) % self.maxsize
        return min_value

    def show(self):
        index = self.head
        flag = True
        while True:
            if index == self.tail:
                if flag:
                    flag = False
                else:
                    break
            print(self.__queue[index], end=" ")
            index = (index + 1) % self.maxsize
        print("")

class CriteriaConfig:
    def __init__(self, criteria: int, **kwargs):
        self.criteria = criteria
        if criteria == 1: # GL准则
            self.GL_alpha = kwargs['GL_alpha']
        elif criteria == 2:
            self.PQ_k = kwargs['PQ_k']
            self.PQ_alpha = kwargs['PQ_alpha']
        elif criteria == 3:
            self.UP_s = kwargs['UP_s']
        else:
            raise ValueError("The arg [criteria] must equals to 1, 2 or 3.")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, config, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.config = config
        self.verbose = verbose
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

        if config.criteria == 2:
            self.que = MyQueue(maxsize=config.PQ_k)
        if config.criteria == 3:
            self.counter = 0

    def _get_GL_loss(self, val_loss):

        return 100*(val_loss / self.val_loss_min - 1)

    def _get_P_loss(self, val_loss):
        P = 1000*(self.que.qsum() / self.config.PQ_k / self.que.qmin() - 1)
        return P
        

    def __call__(self, val_loss, model):

        # score = -val_loss

        # if self.best_score is None:
        #     self.best_score = scorefrom queue import Queue

        #     self.save_checkpoint(val_loss, model)
        # elif score < self.best_score + self.delta:
        #     self.counter += 1
        #     self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        # else:
        #     self.best_score = score
        #     self.save_checkpoint(val_loss, model)
        #     self.counter = 0
        if self.config.criteria == 2:
            self.que.push(val_loss)


        if self.val_loss_min == np.Inf:
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.val_loss_min:
            if self.config.criteria == 1:
                GL_loss = self._get_GL_loss(val_loss)
                self.trace_func(f'>> Validation loss Increased; EarlyStopping criteria 1: GL_loss={GL_loss}; GL_alpha={self.config.GL_alpha}')
                if GL_loss > self.config.GL_alpha:
                    self.early_stop = True
            
            elif self.config.criteria == 2 and self.que.full():
                P_loss = self._get_P_loss(val_loss)
                GL_loss = self._get_GL_loss(val_loss)
                PQ_loss = GL_loss / P_loss
                self.trace_func(f'>> Validation loss Increased; EarlyStopping criteria 2: PQ_loss={PQ_loss}; val_loss_min={self.val_loss_min}')
                if PQ_loss > self.config.PQ_alpha:
                    self.early_stop = True
            elif self.config.criteria == 3:
                self.counter += 1
                self.trace_func(f'>> Validation loss Increased; EarlyStopping criteria 3: UP counter: {self.counter} out of {self.config.UP_s}')
                if self.counter >= self.config.UP_s:
                    self.early_stop = True

        else:
            self.save_checkpoint(val_loss, model)
            if self.config.criteria == 3:
                self.counter = 0
        

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
