package mapreduce

import org.apache.hadoop.conf.{Configurable, Configuration}
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.Partitioner

class PositionPartitioner extends Partitioner[LongWritable, Text] with Configurable {
  private var conf: Configuration = _
  private var positionsPerReducer: Long = _
  private var numReducers: Int = _

  override def setConf(conf: Configuration): Unit = {
    this.conf = conf
    numReducers = conf.getInt("mapreduce.job.reduces", 1)
    val maxPosition = conf.getLong("max.position", Long.MaxValue)
    positionsPerReducer = (maxPosition + numReducers - 1) / numReducers
  }

  override def getConf: Configuration = conf

  override def getPartition(key: LongWritable, value: Text, numPartitions: Int): Int = {
    if (positionsPerReducer == 0) {
      // Fallback in case positionsPerReducer is not set
      positionsPerReducer = Long.MaxValue / numPartitions
    }
    val position = key.get()
    val partition = (position / positionsPerReducer).toInt
    if (partition >= numPartitions) numPartitions - 1 else partition
  }
}
