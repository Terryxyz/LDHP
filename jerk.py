def xlen(lst, start=0, step=1):
    '''用xlen函数直接获得一个列表的索引'''
    return range(start, start + len(lst), step)


#to get the most nearest int from float
#we must use int(round(object<"float">))
#it is too many letters and brackets to need to be coded.
#
#int() and round() can only do with one float number
#they can't do with a list or a dic or a set()
#but xint() can do it
def xint(a):
    '''
    用xint代替int()与round()的组合
    而且xint()可以使一个多维的数组列表、dict、set、tuple中的
    所有的float变成其最接近的整数
    不过不要传入循环dict
    '''

    if isinstance(a, (int, str)):
        #a is int or str
        return a

    elif isinstance(a, float):
        #a is float
        return int(round(a))

    elif isinstance(a, set):
        #a is set
        return {xint(i) for i in a}

    elif isinstance(a, list):
        #a is list
        return [xint(i) for i in a]

    elif isinstance(a, tuple):
        #a is tuple
        return tuple([xint(i) for i in a])

    elif isinstance(a, dict):
        #a is dict
        return {xint(key): xint(value) for key, value in a.iteritems()}


#list can only do with one tuple but not the element in the tuple
#but xlist() can do it
def xlist(a):
    '''
    将所有的tuple元素变成list
    '''

    if isinstance(a, (int, str, float)):
        #a is int or str
        return a

    elif isinstance(a, set):
        #a is set
        return {xlist(i) for i in a}

    elif isinstance(a, list):
        #a is list
        return [xlist(i) for i in a]

    elif isinstance(a, tuple):
        #a is tuple
        return [xlist(i) for i in a]

    elif isinstance(a, dict):
        #a is dict
        return {xlist(key): xlist(value) for key, value in a.iteritems()}


#tuple can only do with one list but not the element in the tuple
#but xtuple() can do it
def xtuple(a):
    '''
    将所有的list元素变成tuple
    '''

    if isinstance(a, (int, str, float)):
        #a is int or str
        return a

    elif isinstance(a, set):
        #a is set
        return {xtuple(i) for i in a}

    elif isinstance(a, list):
        #a is list
        return tuple([xtuple(i) for i in a])

    elif isinstance(a, tuple):
        #a is tuple
        return tuple([xtuple(i) for i in a])

    elif isinstance(a, dict):
        #a is dict
        return {xtuple(key): xtuple(value) for key, value in a.iteritems()}
