import random

def foo(x, y, z):
    return 6*x*3 + y - z +22

def fitness(x, y, z):
    ans = foo(x, y, z)
    
    if ans == 0:
        return 99999
    else:
        return abs(1/ans)
    
    
# Generate solutions
solutions = []
for s in range(1000):
    solutions.append(   (random.uniform(0, 10000),
                         random.uniform(0, 10000),
                         random.uniform(0, 10000)))
    
for i in range(10000):
    
    rankedsolutions = []
    
    for s in solutions:
        rankedsolutions.append( (fitness(s[0], s[1], s[2]), s) )
        
    rankedsolutions.sort()
    rankedsolutions.reverse()
    
    print(f"=== Gen {i} best solution ===")
    print(rankedsolutions[0])
    
    if rankedsolutions[0][0] > 999:
        break
    
    bestsolutions = rankedsolutions[:100]
    
    elements = []
    for s in bestsolutions:
        elements.append(s[1][0])
        elements.append(s[1][1])
        elements.append(s[1][2])
        
    newGen = []
    for _ in range(1000):
        e1 = random.choice(elements) * random.uniform(0.99, 1.01)
        e2 = random.choice(elements) * random.uniform(0.99, 1.01)
        e3 = random.choice(elements) * random.uniform(0.99, 1.01)
        
        newGen.append((e1, e2, e3))
        
    solutions = newGen
        
    
        