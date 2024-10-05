package utility

import scala.io.Source
import java.io.{BufferedWriter, FileWriter, File}

object TextPreprocessor {
  def shardTextFile(inputPath: String, outputDir: String, linesPerShard: Int): Unit = {
    // Open the input file
    val inputFile = new File(inputPath)
    val linesIterator = Source.fromFile(inputFile, "UTF-8").getLines()

    // Create output directory if it doesn't exist
    val outputDirectory = new File(outputDir)
    if (!outputDirectory.exists()) {
      outputDirectory.mkdirs()
    }

    var lineCount = 0
    var shardIndex = 0
    var position = 0L
    var writer: BufferedWriter = null

    try {
      // Initialize the first writer
      writer = new BufferedWriter(new FileWriter(new File(outputDirectory, f"shard_$shardIndex%05d.txt")))

      for (line <- linesIterator) {
        println(s"Processing line ${lineCount + 1} in shard $shardIndex")

        val words = line.split("\\s+").filter(_.nonEmpty)

        words.foreach { word =>
          val outputLine = s"${position}_${word}\n"
          writer.write(outputLine)
          position += 1
        }

        lineCount += 1

        // Check if current shard reached the limit
        if (lineCount >= linesPerShard) {
          println(s"Shard $shardIndex reached $linesPerShard lines. Creating new shard.")
          writer.close()
          shardIndex += 1
          lineCount = 0
          writer = new BufferedWriter(new FileWriter(new File(outputDirectory, f"shard_$shardIndex%05d.txt")))
        }
      }
    } finally {
      // Close the last writer if it's still open
      if (writer != null) {
        writer.close()
      }
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      println("Usage: TextPreprocessor <input path> <output directory> <lines per shard>")
      System.exit(1)
    }

    val inputPath = args(0)
    val outputDir = args(1)
    val linesPerShard = args(2).toInt

    shardTextFile(inputPath, outputDir, linesPerShard)
  }
}
