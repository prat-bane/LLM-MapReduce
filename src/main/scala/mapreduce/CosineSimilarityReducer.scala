package mapreduce

import com.typesafe.config.ConfigFactory
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.Reducer
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.io.Source
import scala.jdk.CollectionConverters.IterableHasAsScala

class CosineSimilarityReducer extends Reducer[Text, Text, Text, Text] {

  private val logger = Logger.getLogger(classOf[CosineSimilarityReducer])
  private val config = ConfigFactory.load()
  var topN: Int = config.getInt("app.cosine-similarity.topN") // Number of top similar words to keep
  var allEmbeddings: Map[String, Array[Double]] = Map() //Map to store embeddings of all words

  override def setup(context: Reducer[Text, Text, Text, Text]#Context): Unit = {
    // Load all embeddings from the Distributed Cache
    val cacheFiles = context.getCacheFiles
    if (cacheFiles != null && cacheFiles.nonEmpty) {
      try {
        // Assuming only one cache file is added
        val cachePath = cacheFiles(0).toString

        // Create a Path object
        val fsPath = new Path(cachePath)
        val fs = fsPath.getFileSystem(context.getConfiguration)
        val stream = fs.open(fsPath)
        val reader = Source.fromInputStream(stream)

        // Parse the embeddings
        val embeddings = reader.getLines().flatMap { line =>
          val parts = line.split("\t")
          if (parts.length == 2) {
            val wordToken = parts(0).trim // e.g., "academic 91356"
            val embeddingStr = parts(1).trim // e.g., "0.123,0.456,0.789"
            val embedding = embeddingStr.split(",").map(_.toDouble)
            if (embedding.nonEmpty) {
              // Extract only the word part, assuming the format "word token"
              val word = wordToken.split(" ")(0) // e.g., "academic"
              Some(word -> embedding)
            } else {
              logger.warn(s"Empty embedding for line: $line")
              None
            }
          } else {
            logger.warn(s"Invalid embedding line format: $line")
            None
          }
        }.toMap

        reader.close()
        allEmbeddings = embeddings
        logger.info(s"Loaded ${allEmbeddings.size} embeddings from the Distributed Cache.")
      } catch {
        case e: Exception =>
          logger.error("Error loading embeddings from the Distributed Cache.", e)
          throw e
      }
    } else {
      logger.error("No embeddings file found in the Distributed Cache.")
      throw new RuntimeException("Embeddings file not found in Distributed Cache.")
    }
  }

  override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = {
    val currentWordToken = key.toString // e.g., "academic 91356"

    // Extract only the word part
    val word = currentWordToken.split(" ")(0) // e.g., "academic"

    // Convert Java Iterable to Scala collection
    values.asScala.foreach { value =>
      val embeddingStr = value.toString.trim // e.g., "0.123,0.456,0.789"
      val currentEmbedding = embeddingStr.split(",").map(_.toDouble)
      logger.debug(s"Embedding for word '$word': $embeddingStr")
      // Compute norm of the current embedding
      val norm1 = math.sqrt(currentEmbedding.map(x => x * x).sum)

      // Compute cosine similarity with all other embeddings
      logger.debug(s"Computing similarities for word '$word'.")
      val similarities = allEmbeddings
        .filter(_._1 != word)
        .map { case (otherWord, otherEmbedding) =>
          val norm2 = math.sqrt(otherEmbedding.map(x => x * x).sum)
          val dotProduct = (currentEmbedding zip otherEmbedding).map { case (x, y) => x * y }.sum
          val cosineSim = if (norm1 != 0.0 && norm2 != 0.0) dotProduct / (norm1 * norm2) else 0.0
          (otherWord, cosineSim)
        }

      // Get top N similar words
      val topSimilar = similarities.toSeq.sortBy(-_._2).take(topN)

      // Format the output: similarWord1:cosineSimilarity similarWord2:cosineSimilarity ...
      val topSimilarStr = topSimilar.map { case (w, s) => s"$w:$s" }.mkString(" ")

      // Combine embedding and similar words
      val outputValueStr = s"$embeddingStr $topSimilarStr"

      // Prepare the output key and value
      val outputKey = new Text(word) // "academic"
      val outputValue = new Text(outputValueStr) // "0.123,0.456,0.789 bow:0.9746318461970762 ours:0.9594119455666708"

      // Emit the result
      context.write(outputKey, outputValue)
      logger.debug(s"Emitted result for word '$word'")
    }
  }
}
