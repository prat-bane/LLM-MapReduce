package mapreduce

import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.Mapper
import org.apache.log4j.Logger

class CosineSimilarityMapper extends Mapper[LongWritable, Text, Text, Text] {

  private val logger = Logger.getLogger(classOf[CosineSimilarityMapper])

  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit = {
    val line = value.toString.trim
    if (line.nonEmpty) {
      try {
        // Split the line by tab to separate word and embedding
        val parts = line.split("\t")
        if (parts.length == 2) {
          val word = parts(0).trim // e.g., "a 64"
          val embeddingStr = parts(1).trim // e.g., "-0.0442546121776104,-0.1295069931074977"

          // Validate embedding format
          val embedding = embeddingStr.split(",").map(_.toDouble)
          if (embedding.nonEmpty) {
            // Emit word as key and embedding as value
            context.write(new Text(word), new Text(embeddingStr))
          } else {
            logger.warn(s"Empty embedding for line: $line")
          }
        } else {
          logger.warn(s"Invalid line format (expected 2 parts separated by tab): $line")
        }
      } catch {
        case e: Exception =>
          logger.error(s"Error processing line: $line", e)
        // Optionally, you can choose to skip or emit the faulty line
      }
    }
  }
}
