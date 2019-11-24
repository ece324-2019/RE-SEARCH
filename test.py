import pyodbc
# import mysql.connector
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
    query = "SELECT TOP 5 UUID,path_to_file FROM classifier.dbo.classified_tops WHERE " + " AND ".join(givelist) + " ORDER BY " + giveparams +" DESC"
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

    for i in range(0,len(addlist)):
        temp = temp + "("
        for j in range(0,len(addlist[i])):
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

# getlist = ['id','path']
# setlist = ["color='red'"]
# givelist = ["color='black'","neckline='turtleneck'"]
# addparams = ['id','color','sleeve','neckline','button','path']
# addlist = [['11','yellow','short','square','no','aritzia'],['12','orange','crew','collar','yes','ssense']]
# write(addparams, addlist)
# L = ['orange','square']
# get_things(L)
# delete()