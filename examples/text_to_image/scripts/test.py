import random
'''
https://blog.csdn.net/u012063773/article/details/79470009
'''
def random_weight(weight_data):
    total = sum(weight_data.values())    # 权重求和
    ra = random.uniform(0, total)   # 在0与权重和之前获取一个随机数 
    curr_sum = 0
    ret = None
    # keys = weight_data.iterkeys()    # 使用Python2.x中的iterkeys
    keys = weight_data.keys()        # 使用Python3.x中的keys
    for k in keys:
        curr_sum += weight_data[k]             # 在遍历中，累加当前权重值
        if ra <= curr_sum:          # 当随机数<=当前权重和时，返回权重key
            ret = k
            break
    return ret
weight_data = {'a': 100, 'b': 15, 'c': 50}
# random_weight(weight_data)

def prompt_content():
    # list_subjects = ['1 girl', '1 boy', '1 girl and 1 cat', '1 boy and 1 cat', '1 girl and 1 dog', '1 boy and 1 cat']
    list_subjects = ['a cat', 'a dog']
    subject = random.choice(list_subjects)
    
    dict_eye_colors = {'brown': 20, 'yellow':20, 'grey':20, 'blue':20, 'green':20, 'red':5}
    eye_color = random_weight(dict_eye_colors)
    eye_style = f'{eye_color} eyes'
    
    dict_hair_colors = {'black': 30, 'brown': 20, 'yellow':10, 'red':10, 'grey':5, 'blue':5, 'green':5, 'pink':5, 'orange':5,
                        'balck and white': 30, 'orange and white':30, 'grey and white': 20}
    hair_color = random_weight(dict_hair_colors)
    hair_style = f'{hair_color} hair'
    r = random.randint(0,5)
    if r == 0:
        hair_style += ', hair clasp'
    elif r == 1:
        hair_style += ', bowknot'
       
    if random.randint(0,3)==0:
         hair_style += ', hat'
    r = random.randint(0,5)
    if r == 0:
        hair_style += ', mask'
    elif r == 1:
        hair_style += ', sunglasses'
    elif r == 2:
        hair_style = ', glass'
      
    
    up_wearing = ['T-shirt', 'shirt', 'jacket', 'sweater', 'vest', 'leather jacket']
    down_wearing = ['skirt', 'pants', 'shorts', 'jeans', 'casual pants']
    all_wearing = ['suit', 'sportswear', 'evening gown', 'trench coat', 'pajamas']
    color_wearing = ['red', 'blue', 'green', 'black', 'white', 'grey', 'yellow', 'pink', 'purple',
                     'orange', 'brown', 'beige', 'gold', 'silver', 'peach', 'burgundy', 'olive green',
                     'sky blue', 'coral', 'teal', 'plaid', 'floral']
  
    if random.randint(0,2):
        up_color = random.choice(color_wearing)
        up = random.choice(up_wearing)
        down_color = random.choice(color_wearing)
        down = random.choice(down_wearing)
        wearing = f'{up_color} {up}, {down_color} {down}'
    else:
        color = random.choice(color_wearing)
        all = random.choice(all_wearing)
        wearing = f'{color} {all}'
        
    list_act = ['sitting', 'standing', 'laying', 'walking', 'running']   
    act = random.choice(list_act)
    dict_body = {'full body':20, 'profile facing the screen':20}
    if random.randint(0,2):
        body = random_weight(dict_body)
        act = f'{act}, {body}'
    
    if random.randint(0,2) and (('sitting' in act) or ('standing' in act) or ('laying' in act)):    #室内
        dict_scenery = {'office':20, 'bedroom':20,  'classroom':20, 'coffee shop':20, 'indoor':20}
        list_scenery = ['floor', 'bed', 'table', 'desk', 'chair', 'lamp', 'cup', 'window', 'sofa']
    elif random.randint(0,2) and (('sitting' in act) or ('standing' in act) or ('walking' in act) or ( 'riding a bicycle' in act)):  #城市户外
        dict_scenery = {'plaza': 20, 'market':5}
        list_scenery = ['blue sky', 'cloud', 'car', 'road', 'tree', 'flower' 'house', 'building']
    else:                                                                                           #自然场景
        dict_scenery = {'farm':10, 'park':20, 'outdoor':20}
        list_scenery = ['lake', 'blue sky', 'cloud', 'grass', 'grassland', 'car', 'road', 'tree', 'flower' 'house']
    scenery = random_weight(dict_scenery)
    item = random.sample(list_scenery, 2)
    scenery = f'{scenery}, {item[0]}, {item[1]}'

    dict_mood = {'happy': 50, 'sad':10, 'angry':10, 'Surprise':20, 'fear':10}
    mood = random_weight(dict_mood)
    
    caption = f'{subject}, {eye_style}, {hair_style}, {wearing}, {act}, {scenery}, {mood}'
    
    return caption
    
for idx in range(50):
    print(f"{idx}: {prompt_content()}")