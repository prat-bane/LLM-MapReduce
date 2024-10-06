// TextPreprocessor.scala
package utility

import scala.io.Source
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import java.io.{BufferedWriter, OutputStreamWriter}

object TextPreprocessorHDFS {
  def shardTextFile(inputPath: String, outputDir: String, linesPerShard: Int): Unit = {
    // Initialize Hadoop Configuration and FileSystem
    val conf = new Configuration()
    val fs = FileSystem.get(conf)

    // Open the input file from HDFS
    val inputPathHDFS = new Path(inputPath)
    val inputStream = fs.open(inputPathHDFS)
    val linesIterator = Source.fromInputStream(inputStream, "UTF-8").getLines()

    // Create output directory in HDFS if it doesn't exist
    val outputPath = new Path(outputDir)
    if (!fs.exists(outputPath)) {
      fs.mkdirs(outputPath)
    }

    var lineCount = 0
    var shardIndex = 0
    var position = 0L
    var writer: BufferedWriter = null

    try {
      // Initialize the first writer
      val shardFilePath = new Path(outputPath, f"shard_$shardIndex%05d.txt")
      val shardOutputStream = fs.create(shardFilePath, true)
      writer = new BufferedWriter(new OutputStreamWriter(shardOutputStream, "UTF-8"))

      for (line <- linesIterator) {

        val words = line.split("\\s+").filter(_.nonEmpty)

        words.foreach { word =>
          val outputLine = s"${position}_${word}\n"
          writer.write(outputLine)
          position += 1
        }

        lineCount += 1

        // Check if current shard reached the limit
        if (lineCount >= linesPerShard) {

          writer.close()
          shardIndex += 1
          lineCount = 0

          // Initialize new writer for next shard
          val newShardFilePath = new Path(outputPath, f"shard_$shardIndex%05d.txt")
          val newShardOutputStream = fs.create(newShardFilePath, true)
          writer = new BufferedWriter(new OutputStreamWriter(newShardOutputStream, "UTF-8"))
        }
      }
    } finally {

      if (writer != null) {
        writer.close()
      }

      inputStream.close()
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length != 3) {

      System.exit(1)
    }

    val inputPath = args(0)
    val outputDir = args(1)
    val linesPerShard = args(2).toInt

    shardTextFile(inputPath, outputDir, linesPerShard)
  }
}
