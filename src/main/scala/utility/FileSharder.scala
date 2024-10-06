package utility

import java.io.{BufferedWriter, File, FileWriter}
import scala.io.Source
import scala.util.matching.Regex
import scala.util.{Failure, Success, Try}

object FileSharder {

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
   * @param outputDirPath Path to the output directory.
   */
  def deleteExistingShards(outputDirPath: String): Unit = {
    val outputDir = new File(outputDirPath)
    if (outputDir.exists() && outputDir.isDirectory) {
      // Define the shard file pattern: starts with "shard_" and ends with ".txt"
      val shardFiles = outputDir.listFiles().filter { file =>
        file.isFile && file.getName.matches("""shard_\d{5}\.txt""")
      }
      shardFiles.foreach { file =>
        if (file.delete()) {
          println(s"Deleted existing shard file: ${file.getName}")
        } else {
          println(s"Failed to delete shard file: ${file.getName}")
        }
      }
    } else {
      // If the directory doesn't exist, create it
      if (!outputDir.exists()) {
        if (outputDir.mkdirs()) {
          println(s"Created output directory: $outputDirPath")
        } else {
          println(s"Failed to create output directory: $outputDirPath")
          throw new RuntimeException(s"Cannot create output directory: $outputDirPath")
        }
      } else {
        throw new IllegalArgumentException(s"Output path is not a directory: $outputDirPath")
      }
    }
  }


  /**
   * Shards a text file into multiple smaller files based on the number of lines per shard.
   *
   * @param inputPath         Path to the input file.
   * @param outputDir         Directory where shard files will be stored.
   * @param linesPerShard     Number of lines per shard.
   * @param skipPreprocessing If true, skips splitting into sentences and preprocessing.
   */
  def shardByLines(
                    inputPath: String,
                    outputDir: String,
                    linesPerShard: Int,
                    skipPreprocessing: Boolean = false
                  ): Unit = {
    require(linesPerShard > 0, "linesPerShard must be a positive integer.")

    val inputFile = new File(inputPath)
    if (!inputFile.exists()) {
      throw new IllegalArgumentException(s"Input file does not exist: $inputPath")
    }

    // Ensure output directory exists
    ensureOutputDirectoryExists(outputDir)

    // Delete existing shards before proceeding
    deleteExistingShards(outputDir)

    Try {
      val source = Source.fromFile(inputFile, "UTF-8")
      try {
        var shardIndex = 1
        var lineCount = 0
        var writer: BufferedWriter = new BufferedWriter(new FileWriter(new File(outputDir, f"shard_$shardIndex%05d.txt")))

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
              writer = new BufferedWriter(new FileWriter(new File(outputDir, f"shard_$shardIndex%05d.txt")))
            }
          }

        writer.close()
      } finally {
        source.close()
      }
    } match {
      case Success(_) =>
        println(s"Successfully created shards in directory: $outputDir")
      case Failure(exception) =>
        println(s"An error occurred during sharding: ${exception.getMessage}")
        exception.printStackTrace()
    }
  }


  /**
   * Ensures that the directory for the given file path exists.
   * If it doesn't, the directory is created.
   *
   * @param filePath The full path to the file.
   */
  def ensureOutputDirectoryExists(filePath: String): Unit = {
    val file = new File(filePath)
    val parentDir = file.getParentFile
    if (parentDir != null && !parentDir.exists()) {
      val created = parentDir.mkdirs()
      if (created) {
        println(s"Created output directory: ${parentDir.getAbsolutePath}")
      } else {
        throw new RuntimeException(s"Failed to create output directory: ${parentDir.getAbsolutePath}")
      }
    }
  }

  /**
   * Extracts token IDs from part-0000x files within the input directory
   * and writes them to the specified output file.
   *
   * @param inputDirPath   Path to the directory containing part-0000x files.
   * @param outputFilePath Path to the output token_ids.txt file.
   */
  def consolidateTokenIds(inputDirPath: String, outputFilePath: String): Unit = {
    // Ensure the output directory exists
    ensureOutputDirectoryExists(outputFilePath)

    // Define regex patterns
    val partFileRegex: Regex = """^part-r-\d{5}$""".r
    val tokenExtractRegex: Regex = """\[(\d+(?:\s+\d+)*)\]""".r

    // Initialize the output file with a BufferedWriter
    val bw: BufferedWriter = new BufferedWriter(new FileWriter(outputFilePath, false)) // 'false' to overwrite
    try {
      val inputDir = new File(inputDirPath)
      if (!inputDir.exists() || !inputDir.isDirectory) {
        throw new IllegalArgumentException(s"Input directory does not exist or is not a directory: $inputDirPath")
      }

      // List all files in the input directory matching part-0000x
      val partFiles = inputDir.listFiles().filter { file =>
        partFileRegex.pattern.matcher(file.getName).matches()
      }

      if (partFiles.isEmpty) {
        println(s"No part-0000x files found in directory: $inputDirPath")
      } else {
        println(s"Found ${partFiles.length} part-0000x files. Starting extraction...")

        // Process each part file
        partFiles.foreach { file =>
          println(s"Processing file: ${file.getName}")
          val source = Source.fromFile(file, "UTF-8")
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
                  } // Append tokens with a space
                case None =>
                // Line does not contain tokens within brackets; skip or log
              }
            }
          } catch {
            case e: Exception =>
              println(s"Error processing file ${file.getName}: ${e.getMessage}")
          } finally {
            source.close()
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
        case Success(_) => println(s"Tokens successfully written to $outputFilePath")
        case Failure(ex) => println(s"Failed to close the output file: ${ex.getMessage}")
      }
    }
  }
  /**
   * Entry point of the application.
   *
   * @param args Command-line arguments.
   *             args(0): Input file path
   *             args(1): Output directory path
   *             args(2): Number of lines per shard
   */
  def main(args: Array[String]): Unit = {
    /*if (args.length != 3) {
      println(
        """
          |Usage:
          |  FileSharder <inputFilePath> <outputDirPath> <linesPerShard>
          |
          |Example:
          |  FileSharder /path/to/input.txt /path/to/output 1000
          |""".stripMargin)
      System.exit(1)
    }*/
    shardByLines("E:\\wikitext_test.txt",
      "E:\\output\\text\\shards",
      1000,false)
   /* val Array(inputPath, outputDir, linesStr) = args
    val linesPerShard = Try(linesStr.toInt).getOrElse {
      println("linesPerShard must be a valid integer.")
      System.exit(1)
      0 // Unreachable
    }*/
   /* shardByLines("D:\\input.txt","D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\text\\shards\\",
      100,false)*/
    /*shardByLines("D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\input\\tokenids.txt",
      "D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\input", 200,true)*/
    /*shardByLines("D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\embeddings\\output\\part-r-00000",
      "D:\\IdeaProjects\\ScalaRest\\src\\main\\resources\\mapreduce\\similarity\\shards",200,true)*/
  }
}
