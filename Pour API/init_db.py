import sqlite3

connection = sqlite3.connect('database.db')


with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO posts (title, SourceIP, DestinationIP, SourcePort,DestinationPort, Duration , FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate,PacketLengthVariance, PacketLengthStandardDeviation, PacketLengthMean, PacketLengthMedian, PacketLengthMode, PacketLengthSkewFromMedian,PacketLengthSkewFromMode, PacketLengthCoefficientofVariation, PacketTimeVariance, PacketTimeStandardDeviation, PacketTimeMean, PacketTimeMedian, PacketTimeMode,PacketTimeSkewFromMedian, PacketTimeSkewFromMode, PacketTimeCoefficientofVariation, ResponseTimeTimeVariance, ResponseTimeTimeStandardDeviation,ResponseTimeTimeMean, ResponseTimeTimeMedian, ResponseTimeTimeMode, ResponseTimeTimeSkewFromMedian, ResponseTimeTimeSkewFromMode, ResponseTimeTimeCoefficientofVariation) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?)",
            ('First Post', 34567,654321,443,34454,120.01754299999999,378423,	3153.0640483116713,	200090,	1667.172939875965,	18017.6813248642,	134.2299568831943,	176.69914477703117,	173.0,	68,	0.08267479621371245,	0.8097979564399338,	0.7596525554924056,	1197.3148020755832,	34.60223695190216,	59.41828608949298,	58.89281,	0.0,	0.045558565206931384,	1.7171804866860385,	0.5823499671428745,	1.5384481516131536e-09,	3.9223056377762736e-05,	4.4756723716381416e-05,	0,	0,	0.2108497377025665,	0.12127366288258473,	0.8763612061131877)
            )

cur.execute("INSERT INTO posts (title, SourceIP, DestinationIP, SourcePort,DestinationPort, Duration , FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate,PacketLengthVariance, PacketLengthStandardDeviation, PacketLengthMean, PacketLengthMedian, PacketLengthMode, PacketLengthSkewFromMedian,PacketLengthSkewFromMode, PacketLengthCoefficientofVariation, PacketTimeVariance, PacketTimeStandardDeviation, PacketTimeMean, PacketTimeMedian, PacketTimeMode,PacketTimeSkewFromMedian, PacketTimeSkewFromMode, PacketTimeCoefficientofVariation, ResponseTimeTimeVariance, ResponseTimeTimeStandardDeviation,ResponseTimeTimeMean, ResponseTimeTimeMedian, ResponseTimeTimeMode, ResponseTimeTimeSkewFromMedian, ResponseTimeTimeSkewFromMode, ResponseTimeTimeCoefficientofVariation) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?)",
            ('Second Post', 34567,654321,443,34454,120.01754299999999,378423,	3153.0640483116713,	200090,	1667.172939875965,	18017.6813248642,	134.2299568831943,	176.69914477703117,	173.0,	68,	0.08267479621371245,	0.8097979564399338,	0.7596525554924056,	1197.3148020755832,	34.60223695190216,	59.41828608949298,	58.89281,	0.0,	0.045558565206931384,	1.7171804866860385,	0.5823499671428745,	1.5384481516131536e-09,	3.9223056377762736e-05,	4.4756723716381416e-05,	0,	0,	0.2108497377025665,	0.12127366288258473,	0.8763612061131877)
            )

connection.commit()
connection.close()


