import pyodbc
table = "classifier.dbo.classified_tops"
conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=DESKTOP-SG3RI9Q;'
                          'Database=classifier;'
                          'Trusted_Connection=yes')
cursor = conn.cursor()

def get_things(L):
    givelist,giveparams = make_givelist(L)
    # print(givelist)
    # print(giveparams)
    query = "SELECT TOP 10 UUID,path_to_file FROM classifier.dbo.classified_tops WHERE " + " AND ".join(givelist) + " ORDER BY " + giveparams +" DESC"
    # query = "SELECT TOP 5 UUID,path FROM test.dbo.classified_tops ORDER BY color_cert DESC"
    print(query)
    cursor.execute(query)

    out = []
    for row in cursor:
        out += [[row[0],row[1]]]
        print(row)

    return out
def delete():
    cursor.execute('DELETE FROM classifier.dbo.classified_tops')
    conn.commit()
def write(addparams,addlist):
    temp = ""
    idx = -1
    # print(addlist)
    for i in range(0,len(addlist)):
        temp = temp + "("
        for j in range(0,len(addlist[0])):
            if j == 2 or j == 4 or j == 6 or j == 8 or addlist[i][j] == 'NULL':
                temp = temp + addlist[i][j]
            else:
                temp = temp + "'" + addlist[i][j] + "'"

            if j != len(addlist[i])-1:
                temp = temp + ","
        if i == len(addlist)-1:
            temp = temp + ")"
        else:
            temp = temp + "), "

    query = "INSERT INTO " + table + " (" + ",".join(addparams) + ") VALUES " + temp
    # query = "INSERT INTO test.dbo.classified_tops (UUID,color,color_cert,neckline,neckline_cert,sleeves,sleeves_cert,buttons,buttons_cert,path) VALUES (10,'black',0.9,'NULL',0,'NULL',0,'NULL',0,'../test_folder')"
    print(query)
    cursor.execute(query)
    conn.commit()

def update(setlist, givelist):
    query = "UPDATE " + table + " SET "+", ".join(setlist) + " WHERE " + " AND ".join(givelist)
    print("query: ", query)
    cursor.execute(query)
    conn.commit()

def readall():
    cursor.execute('SELECT * FROM classifier.dbo.classified_tops')
    for row in cursor:
        for i in range(0,len(row)):
            if row[i] == None:
                row[i] = '          '
        print("id=",row[0],
              "color=", row[1],
              "sleeve length=", row[2],
              "neckline=", row[3],
              "buttons=", row[4],
              "path=", row[5])

def make_givelist(L):
    arg = []
    params = ""
    type = [['color=',[['black'],['blue','cyan','ocean'],['red','pomegranate'],['green','forest','olive'],['yellow','egg','sun'],['white','beige','neutral'],['orange','sunset']]],
            ['neckline=', [["crew","round","scoop"], ["square","square-neck","square-necks","flat"], ["turtle","turtleneck","turtlenecks"], ["v-neck","v"]]],
            ['sleeves=', [["long"], ["short"], ["sleeveless"]]],
            ['buttons=',[["buttons"]]]]

    for i in range(0,len(L)):
        for j in range(0,len(type)):
            for k in range(0,len(type[j][1])):
                for a in range(0,len(type[j][1][k])):
                    if L[i] == type[j][1][k][a]:
                        temp = type[j][0] + "'" + type[j][1][k][0] + "'"
                        params += type[j][0][0:-1] + '_confidence' + ' * '
                        arg += [temp]
    params = params[0:-3]
    return arg,params