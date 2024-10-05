package mapreduce

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileStatus, FileSystem, Path}
import org.apache.hadoop.io._
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, TextInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, TextOutputFormat}
import utility.FileSharderHdfs

import scala.io.Source

object TokenizerJob {
  def main(args: Array[String]): Unit = {
    if (args.length != 6) {
      println("Usage: TokenizerJob <input path> <output path> <number of reducers>")
      System.exit(-1)
    }

    val conf = new Configuration()
    val fs = FileSystem.get(conf)

    val inputPath = new Path(args(0))
    val outputPath = new Path(args(1))
    val numReducers = args(2).toInt

    // Preprocess to find the maximum position
    val maxPosition = getMaxPosition(fs, inputPath)
    conf.setLong("max.position", maxPosition)
    conf.setInt("mapreduce.job.reduces", numReducers)

    val job = Job.getInstance(conf, "Tokenize Words with Positions")
    job.setJarByClass(this.getClass)

    // Set Mapper and Reducer classes
    job.setMapperClass(classOf[TokenizerMapper])
    job.setReducerClass(classOf[TokenizerReducer])

    // Set output key and value types
    job.setMapOutputKeyClass(classOf[LongWritable])
    job.setMapOutputValueClass(classOf[Text])
    job.setOutputKeyClass(classOf[NullWritable])
    job.setOutputValueClass(classOf[Text])

    // Set input and output formats
    job.setInputFormatClass(classOf[TextInputFormat])
    job.setOutputFormatClass(classOf[TextOutputFormat[NullWritable, Text]])

    // Set custom partitioner
    job.setPartitionerClass(classOf[PositionPartitioner])
    // Since our partitioner implements Configurable, Hadoop will automatically call setConf on it

    // Set input and output paths using FileInputFormat and FileOutputFormat
    FileInputFormat.addInputPath(job, inputPath)
    FileOutputFormat.setOutputPath(job, outputPath)

    // Submit the job
    val success = job.waitForCompletion(true)

    if(success){
      if(job.waitForCompletion(true)) {
        FileSharderHdfs.consolidateTokenIds(args(1),args(3))
        FileSharderHdfs.shardByLines(args(3),
          args(4),
          args(5).toInt,true)
      } else
        System.exit(1)
    }
    System.exit(if (success) 0 else 1)
  }

  def getMaxPosition(fs: FileSystem, inputPath: Path): Long = {
    var maxPosition = 0L
    val statusList = fs.listStatus(inputPath)
    val files = getAllFiles(statusList)

    for (file <- files) {
      val stream = fs.open(file)
      val reader = Source.fromInputStream(stream)
      for (line <- reader.getLines()) {
        val splitIndex = line.indexOf('_')
        if (splitIndex > 0 && splitIndex < line.length - 1) {
          val positionStr = line.substring(0, splitIndex)
          try {
            val position = positionStr.toLong
            if (position > maxPosition) {
              maxPosition = position
            }
          } catch {
            case _: NumberFormatException =>
          }
        }
      }
      reader.close()
    }
    maxPosition
  }

  def getAllFiles(statusList: Array[FileStatus]): Array[Path] = {
    statusList.flatMap { status =>
      if (status.isDirectory) {
        val fs = status.getPath.getFileSystem(new Configuration())
        getAllFiles(fs.listStatus(status.getPath))
      } else {
        Array(status.getPath)
      }
    }
  }
}

