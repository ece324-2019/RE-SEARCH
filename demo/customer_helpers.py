import pyodbc
table = "classifier.dbo.classified_tops"
conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=DESKTOP-SG3RI9Q;'
                          'Database=classifier;'
                          'Trusted_Connection=yes')
cursor = conn.cursor()

def get_things(L):
    givelist,giveparams = make_givelist(L)
    if not givelist:
        return []
    query = "SELECT TOP 10 UUID,path_to_file FROM classifier.dbo.classified_tops WHERE " + " AND ".join(givelist) + " ORDER BY " + giveparams +" DESC"
    # query = "SELECT TOP 5 UUID,path FROM test.dbo.classified_tops ORDER BY color_cert DESC"
    # print(query)
    cursor.execute(query)

    out = []
    for row in cursor:
        out += [[row[0],row[1]]]
        # print(row)

    return out

def make_givelist(L):
    arg = []
    params = ""
    type = [['color=',['black','blue','red','green','yellow','white','orange']],
            ['neckline=', ["collar", "crew", "square", "turtleneck", "v-neck"]],
            ['sleeves=', ["long", "short", "sleeveless"]]]
    for i in range(0,len(L)):
        for j in range(0,len(type)):
            for k in range(0,len(type[j][1])):
                if L[i] == type[j][1][k]:
                    temp = type[j][0] + "'" + L[i] + "'"
                    params += type[j][0][0:-1] + '_confidence' + ' + '
                    arg += [temp]
    params = params[0:-3]
    return arg,params