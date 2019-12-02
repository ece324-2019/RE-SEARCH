import pyodbc
""" This file contains the helper functions reqired to generate queries to search the database with specified parameters inputted from customer.py"""
""" Server = 'YOUR LOCAL DATABASE SERVER', Database = 'THE NAME OF YOUR LOCAL DATABASE', table = 'THE NAME OF THE TABLE IN YOUR LOCAL DATABASE' """

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
    cursor.execute(query)

    out = []
    for row in cursor:
        out += [[row[0],row[1]]]
        # print(row)

    return out

def make_givelist(L):
    arg = []
    params = ""
    type = [['color=',[['black'],['blue','cyan','ocean'],['red','pomegranate'],['green','forest','olive'],['yellow','egg','sun'],['white','beige','neutral'],['orange','sunset']]],
            ['neckline=', [["crew","round","scoop"], ["square","square-neck","square-necks","flat"], ["turtle","turtleneck","turtlenecks"], ["v-neck","v"]]],
            ['sleeves=', [["long"], ["short"], ["sleeveless"]]],
            ['buttons=',[["buttons", 'button']]]]

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