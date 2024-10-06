package utility

import org.apache.hadoop.fs.{LocatedFileStatus, RemoteIterator}

import java.io.{BufferedWriter, OutputStreamWriter}
import scala.io.Source
import scala.util.matching.Regex
import scala.util.{Failure, Success, Try}

// Import Hadoop FileSystem classes
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}

object FileSharderHdfs {

  /**
   * Preprocesses a sentence by replacing punctuation (excluding apostrophes) with space,
   * collapsing multiple spaces, trimming, and converting to lowercase.
   *
   * @param sentence The input sentence.
   * @return The preprocessed sentence.
   */
  def preprocessSentence(sentence: String): String =
    sentence
      .replaceAll("""[^\w\s']""", " ") // Replace punctuation except apostrophes with space
      .replaceAll("\\s+", " ")        // Collapse multiple spaces into one
      .trim.toLowerCase               // Trim leading/trailing spaces and convert to lowercase

  /**
   * Splits text into sentences based on punctuation marks (. ! ?).
   *
   * @param text The input text.
   * @return A sequence of sentences.
   */
  def splitIntoSentences(text: String): Seq[String] =
    text.split("(?<=[.!?])\\s+").filter(_.nonEmpty)

  /**
   * Deletes existing shard files in the output directory.
   * Shard files are identified by the pattern "shard_XXXXX.txt".
   *
   * @param fs             The Hadoop FileSystem object.
   * @param outputDirPath  Path to the output directory.
   */
  def deleteExistingShards(fs: FileSystem, outputDirPath: Path): Unit = {
    if (fs.exists(outputDirPath) && fs.isDirectory(outputDirPath)) {
      // Define the shard file pattern: starts with "shard_" and ends with ".txt"
      val shardFiles = fs.listStatus(outputDirPath).filter { status =>
        status.isFile && status.getPath.getName.matches("""shard_\d{5}\.txt""")
      }
      shardFiles.foreach { status =>
        if (fs.delete(status.getPath, false)) {
          println(s"Deleted existing shard file: ${status.getPath.getName}")
        } else {
          println(s"Failed to delete shard file: ${status.getPath.getName}")
        }
      }
    } else {
      // If the directory doesn't exist, create it
      if (!fs.exists(outputDirPath)) {
        if (fs.mkdirs(outputDirPath)) {
          println(s"Created output directory: ${outputDirPath.toString}")
        } else {
          println(s"Failed to create output directory: ${outputDirPath.toString}")
          throw new RuntimeException(s"Cannot create output directory: ${outputDirPath.toString}")
        }
      } else {
        throw new IllegalArgumentException(s"Output path is not a directory: ${outputDirPath.toString}")
      }
    }
  }

  /**
   * Shards a text file into multiple smaller files based on the number of lines per shard.
   *
   * @param inputPathString    Path to the input file.
   * @param outputDirString    Directory where shard files will be stored.
   * @param linesPerShard      Number of lines per shard.
   * @param skipPreprocessing  If true, skips splitting into sentences and preprocessing.
   */
  def shardByLines(
                    inputPathString: String,
                    outputDirString: String,
                    linesPerShard: Int,
                    skipPreprocessing: Boolean = false
                  ): Unit = {

    require(linesPerShard > 0, "linesPerShard must be a positive integer.")

    // Initialize Hadoop Configuration and FileSystem
    val conf = new Configuration()
    val fs = FileSystem.get(conf)

    val inputPath = new Path(inputPathString)
    val outputDir = new Path(outputDirString)

    if (!fs.exists(inputPath)) {
      throw new IllegalArgumentException(s"Input file does not exist: $inputPathString")
    }

    // Ensure output directory exists
    ensureOutputDirectoryExists(fs, outputDir)

    // Delete existing shards before proceeding
    deleteExistingShards(fs, outputDir)

    Try {
      val inputStream = fs.open(inputPath)
      val source = Source.fromInputStream(inputStream, "UTF-8")
      try {
        var shardIndex = 1
        var lineCount = 0

        def createWriter(shardIndex: Int): BufferedWriter = {
          val shardPath = new Path(outputDir, f"shard_$shardIndex%05d.txt")
          val outputStream = fs.create(shardPath, true) // Overwrite if exists
          new BufferedWriter(new OutputStreamWriter(outputStream, "UTF-8"))
        }

        var writer: BufferedWriter = createWriter(shardIndex)

        // Process each line using functional constructs
        source.getLines()
          .flatMap { line =>
            if (!skipPreprocessing) {
              // Split line into sentences and preprocess each sentence
              splitIntoSentences(line)
                .map(preprocessSentence)
                .filter(_.nonEmpty)
            } else {
              // Treat each line as a token
              Some(line.trim).filter(_.nonEmpty)
            }
          }
          .foreach { token =>
            writer.write(token)
            writer.newLine()
            lineCount += 1

            // Check if current shard reached the limit
            if (lineCount >= linesPerShard) {
              writer.close()
              shardIndex += 1
              lineCount = 0
              writer = createWriter(shardIndex)
            }
          }

        writer.close()
      } finally {
        source.close()
        inputStream.close()
      }
    } match {
      case Success(_) =>
        println(s"Successfully created shards in directory: $outputDirString")
      case Failure(exception) =>
        println(s"An error occurred during sharding: ${exception.getMessage}")
        exception.printStackTrace()
    }
  }

  /**
   * Ensures that the directory for the given path exists.
   * If it doesn't, the directory is created.
   *
   * @param fs      The Hadoop FileSystem object.
   * @param dirPath The directory path as a Hadoop Path object.
   */
  def ensureOutputDirectoryExists(fs: FileSystem, dirPath: Path): Unit = {
    if (!fs.exists(dirPath)) {
      val created = fs.mkdirs(dirPath)
      if (created) {
        println(s"Created output directory: ${dirPath.toString}")
      } else {
        throw new RuntimeException(s"Failed to create output directory: ${dirPath.toString}")
      }
    }
  }

  /**
   * Extracts token IDs from part-0000x files within the input directory
   * and writes them to the specified output file.
   *
   * @param inputDirPathString   Path to the directory containing part-0000x files.
   * @param outputFilePathString Path to the output token_ids.txt file.
   */
  def consolidateTokenIds(inputDirPathString: String, outputFilePathString: String): Unit = {
    val conf = new Configuration()
    val fs = FileSystem.get(conf)

    val inputDirPath = new Path(inputDirPathString)
    val outputFilePath = new Path(outputFilePathString)

    // Ensure the output directory exists
    ensureOutputDirectoryExists(fs, outputFilePath.getParent)

    // Define regex patterns
    val partFileRegex: Regex = """^part-r-\d{5}$""".r
    val tokenExtractRegex: Regex = """\[(\d+(?:\s+\d+)*)\]""".r

    // Initialize the output file with a BufferedWriter
    val outputStream = fs.create(outputFilePath, true) // 'true' to overwrite
    val bw: BufferedWriter = new BufferedWriter(new OutputStreamWriter(outputStream, "UTF-8"))

    try {
      if (!fs.exists(inputDirPath) || !fs.isDirectory(inputDirPath)) {
        throw new IllegalArgumentException(s"Input directory does not exist or is not a directory: $inputDirPathString")
      }

      // List all files in the input directory matching part-0000x
      val partFiles = fs.listStatus(inputDirPath).filter { status =>
        partFileRegex.pattern.matcher(status.getPath.getName).matches()
      }

      if (partFiles.isEmpty) {
        println(s"No part-0000x files found in directory: $inputDirPathString")
      } else {
        println(s"Found ${partFiles.length} part-0000x files. Starting extraction...")

        // Process each part file
        partFiles.foreach { status =>
          val file = status.getPath
          println(s"Processing file: ${file.getName}")
          val inputStream = fs.open(file)
          val source = Source.fromInputStream(inputStream, "UTF-8")
          try {
            for (line <- source.getLines()) {
              // Extract tokens within brackets
              tokenExtractRegex.findFirstMatchIn(line) match {
                case Some(m) =>
                  val tokensStr = m.group(1) // Extracted string of tokens
                  val tokens = tokensStr.trim.split("\\s+")
                  tokens.foreach { token =>
                    bw.write(token)
                    bw.newLine()
                  }
                case None =>
                // Line does not contain tokens within brackets; skip or log
              }
            }
          } catch {
            case e: Exception =>
              println(s"Error processing file ${file.getName}: ${e.getMessage}")
          } finally {
            source.close()
            inputStream.close()
          }
        }

        println("Token extraction completed.")
      }
    } catch {
      case e: Exception =>
        println(s"An error occurred: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      // Ensure the BufferedWriter is closed
      Try(bw.close()) match {
        case Success(_) => println(s"Tokens successfully written to $outputFilePathString")
        case Failure(ex) => println(s"Failed to close the output file: ${ex.getMessage}")
      }
    }
  }

  def findFilePath(hdfsUri: String, searchPath: String, fileNameToFind: String): Option[String] = {
    // Initialize Hadoop Configuration
    val conf = new Configuration()

    // Get the FileSystem instance
    val fs = FileSystem.get(conf)

    try {
      val dirPath = new Path(searchPath)

      // Validate if the search path exists and is a directory
      if (!fs.exists(dirPath)) {
        println(s"Directory does not exist: $searchPath")
        return None
      }

      if (!fs.isDirectory(dirPath)) {
        println(s"The provided search path is not a directory: $searchPath")
        return None
      }

      // Initialize an iterator to list files recursively
      val fileIterator: RemoteIterator[LocatedFileStatus] = fs.listFiles(dirPath, true)

      // Iterate through the files and find the desired file
      while (fileIterator.hasNext) {
        val fileStatus: LocatedFileStatus = fileIterator.next()
        val filePath: Path = fileStatus.getPath

        if (filePath.getName.equals(fileNameToFind)) {
          // File found; return its absolute path
          return Some(filePath.toString)
        }
      }

      // File not found
      None
    } catch {
      case e: Exception =>
        println("An error occurred while searching for the file:")
        e.printStackTrace()
        None
    } finally {
      // Ensure FileSystem is closed to free resources
      fs.close()
    }
  }

  /**
   * Entry point of the application.
   *
   * @param args Command-line arguments.
   *             args(0): Input file path
   *             args(1): Output directory path
   *             args(2): Number of lines per shard
   *             args(3): Skip preprocessing (true/false)
   */
  def main(args: Array[String]): Unit = {
    if (args.length != 4) {

      System.exit(1)
    }

    val Array(inputPath, outputDir, linesStr, skipPreprocessingStr) = args
    val linesPerShard = Try(linesStr.toInt).getOrElse {
      println("linesPerShard must be a valid integer.")
      System.exit(1)
      0 // Unreachable
    }
    val skipPreprocessing = Try(skipPreprocessingStr.toBoolean).getOrElse {
      println("skipPreprocessing must be a valid boolean (true or false).")
      System.exit(1)
      false // Unreachable
    }

    shardByLines(inputPath, outputDir, linesPerShard, skipPreprocessing)
  }
}
