package mapreduce

import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, FileSplit}
import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.{InputSplit, JobContext, RecordReader, TaskAttemptContext}

class WholeFileInputFormat extends FileInputFormat[Text, Text] {
  override def isSplitable(context: JobContext, filename: Path): Boolean = false

  override def createRecordReader(
                                   split: InputSplit,
                                   context: TaskAttemptContext
                                 ): RecordReader[Text, Text] = {
    new WholeFileRecordReader()
  }
}

class WholeFileRecordReader extends RecordReader[Text, Text] {
  private var processed = false
  private var key: Text = new Text()
  private var value: Text = new Text()
  private var fileSplit: FileSplit = _
  private var conf: Configuration = _

  override def initialize(split: InputSplit, context: TaskAttemptContext): Unit = {
    this.fileSplit = split.asInstanceOf[FileSplit]
    this.conf = context.getConfiguration
  }

  override def nextKeyValue(): Boolean = {
    if (!processed) {
      val filePath = fileSplit.getPath
      val fs = filePath.getFileSystem(conf)
      val fileContent = fs.open(filePath)

      // Read the file line by line to avoid loading entire file into memory
      val reader = new java.io.BufferedReader(new java.io.InputStreamReader(fileContent, "UTF-8"))
      val contentBuilder = new StringBuilder()

      Stream.continually(reader.readLine()).takeWhile(_ != null).foreach { line =>
        contentBuilder.append(line).append("\n")
      }

      reader.close()
      fileContent.close()

      key.set(filePath.toString)
      value.set(contentBuilder.toString())

      processed = true
      true
    } else {
      false
    }
  }

  override def getCurrentKey: Text = key

  override def getCurrentValue: Text = value

  override def getProgress: Float = if (processed) 1.0f else 0.0f

  override def close(): Unit = {}
}
