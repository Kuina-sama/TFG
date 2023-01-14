import matplotlib.pyplot as plt

#######################################################################################

# CONSTANTS

#######################################################################################
text_to_num = {'to':{'PARTNER:female':0,'PARTNER:male':1,"PARTNER:unknown":2},
                'as':{'SELF:female':0, 'SELF:male':1,'SELF:unknown':2},
                'about':{'ABOUT:female':0,'ABOUT:male':1,'ABOUT:unknown':2}}

num_to_text = {'to':{0:'PARTNER:female',1:'PARTNER:male',2:"PARTNER:unknown"},
                'as':{0:'SELF:female', 1:'SELF:male',2:'SELF:unknown'},
                'about':{0:'ABOUT:female',1:'ABOUT:male',2:'ABOUT:unknown'}}

all_tasks_names = ['about','as','to']


task_to_num =  {'about':0,'as':1,'to':2}

num_to_task = {0:'about',1:'as',2:'to'}



#######################################################################################

# GENERIC FUNCTIONS

#######################################################################################
def plot_losses_val(train_loss,val_loss):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.legend(['train loss','validation loss'])
    plt.title('Train-Validation loss')
    plt.show()

    return

def plot_losses_train(train_loss):
    plt.plot(train_loss)
    plt.legend(['train loss'])
    plt.title('Train loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    return



