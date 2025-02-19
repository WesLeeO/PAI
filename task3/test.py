import numpy as np

def main():
    n = np.array([[0,10]])
    print(n)
    print(n.shape)
    print(n[:,0])
    x0 = n[:, 0] + (n[:, 1] - n[:, 0]) * \
                 np.random.rand(n.shape[0])
    print(x0)
    print(x0.shape)

    print(np.linspace(0,10,20))





#if oyu run as a script
if __name__ == "__main__":
    main()

  # point = self.optimize_acquisition_function()
        #mean, std = self.v.predict(np.atleast_2d(point), return_std=True)
       # if mean >= 0:
            #print("should not happen")
            #print(mean)
            #self.recommend_next()
       # return np.array(point)
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        #point = self.optimize_acquisition_function()
        #print("seriously ", self.v.predict(np.atleast_2d(point), return_std=True))
        
       # print('??')
        
      #  point = self.optimize_acquisition_function()
      #  m, s = self.v.predict(np.atleast_2d(point), return_std=True) 
       # print(f'Entered with {m}')
       # if(m > 0):
       #    print('fallback')
      #     safe_inputs = [input for i, input in enumerate(self.inputs) if self.v_values[i] < self.max_sa]
      #     #return sum(safe_inputs) / len(safe_inputs)
      #     return np.array(random.sample(safe_inputs, 1)[0])
      #  return np.array(point)
        """
        new_safe_set = self.safe_set.copy()
        for safe_point in self.safe_set:
            mean, std = self.v.predict(np.atleast_2d(safe_point), return_std=True)
            lower_bound = mean - std
            for point in self.D:
                if abs(point - safe_point) <= lower_bound / self.L:
                    new_safe_set.add(point)
        """
       # print("New safe set", len(new_safe_set))            
       # difference = set(self.D).difference(new_safe_set)
        """
        G = set() #extend safe set
        M = set() #find potential optimizers
      
        for p1 in new_safe_set:
            mean, std = self.v.predict(np.atleast_2d(p1), return_std=True)
            upper_bound = mean + std
            for p2 in difference:
                if abs(p2 - p1) <= upper_bound / self.L:
                    G.add(p1)
                    break
        maximum = float('-inf')
      #  print('where1')
        d = {}
        for p1 in new_safe_set:
            mean, std = self.f.predict(np.atleast_2d(p1), return_std=True)
            d[p1] = (mean, std)
            lower_bound = mean - std
            if lower_bound > maximum:
                maximum = lower_bound
        for p1 in new_safe_set:
            upper_bound = d[p1][0] + d[p1][1]
            if upper_bound >= maximum:
                M.add(p1)  
       #print('where2')
        #print('??', len(G), len(M))
        union = G.union(M)
        arg = None
        max_std = float('-inf')
       # print(max_std)
       # print(len(union))
        for point in union:
            std = d[point][1]
            #print(std)
            if std > max_std:
              #  print('should be reached')
                max_std = std
                arg = point
        self.safe_set = new_safe_set
       # print('where3')
      #  print(arg)
        return np.array(arg)
        """
    
        
        

     


     