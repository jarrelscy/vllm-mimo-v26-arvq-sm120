"""Early LR reduction and low-LR patience, based only on fixed validation."""
class AdaptiveSchedule:
    def __init__(self,initial,scheduled_drop=45,relative_worsening=.001,checks=3,patience_updates=15):
        self.best=initial;self.drop_after=scheduled_drop;self.relative_worsening=relative_worsening;self.checks=checks;self.patience_updates=patience_updates;self.bad_checks=0;self.last_best=0;self.early_drop=False
    def observe(self,step,value):
        if value<self.best:self.best=value;self.last_best=step
        if step<=self.drop_after:
            self.bad_checks=self.bad_checks+1 if value>self.best*(1+self.relative_worsening) else 0
            if self.bad_checks>=self.checks and step<self.drop_after:self.drop_after=step;self.early_drop=True
        return step>=self.drop_after+self.patience_updates and step-max(self.drop_after,self.last_best)>=self.patience_updates
