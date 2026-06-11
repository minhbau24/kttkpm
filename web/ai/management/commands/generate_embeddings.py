from django.core.management.base import BaseCommand
from catalog.models import Product
from ai.models import ProductEmbedding
from ai.services import EmbeddingService

class Command(BaseCommand):
    help = "Generates text embeddings for all products in the database in batches and saves them to the ProductEmbedding table."

    def add_arguments(self, parser):
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite existing product embeddings',
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit the number of products to process',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=256,
            help='Batch size for sentence embedding encoding',
        )

    def handle(self, *args, **options):
        overwrite = options['overwrite']
        limit = options['limit']
        batch_size = options['batch_size']

        products = Product.objects.all()
        if not overwrite:
            # Optimize: exclude products that already have embeddings at the DB level
            existing_ids = ProductEmbedding.objects.values_list('product_id', flat=True)
            products = products.exclude(id__in=existing_ids)

        if limit:
            products = products[:limit]

        total = products.count()
        if total == 0:
            self.stdout.write(self.style.SUCCESS("All products already have embeddings. Nothing to do!"))
            return

        self.stdout.write(self.style.SUCCESS(f"Found {total} products that need embeddings. Loading Sentence Transformer model..."))
        
        # Load model
        model = EmbeddingService.get_model()
        
        products_list = list(products)
        self.stdout.write(f"Encoding descriptions and saving to database in batches of {batch_size}...")

        success_count = 0
        for i in range(0, total, batch_size):
            batch_products = products_list[i : i + batch_size]
            
            # Generate texts
            texts = [EmbeddingService.generate_text_representation(p) for p in batch_products]
            
            try:
                # Batch encode using PyTorch under the hood
                vectors = model.encode(texts, batch_size=len(texts), show_progress_bar=False)
                
                # Bulk create or update
                embeddings_to_create = []
                for product, vector in zip(batch_products, vectors):
                    # If overwrite, delete first
                    if overwrite:
                        ProductEmbedding.objects.filter(product=product).delete()
                    embeddings_to_create.append(
                        ProductEmbedding(product=product, embedding_vector=vector.tolist())
                    )
                
                ProductEmbedding.objects.bulk_create(embeddings_to_create)
                success_count += len(batch_products)
                self.stdout.write(f"Processed and saved {success_count}/{total} products...")
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error encoding batch starting at index {i}: {str(e)}"))

        self.stdout.write(
            self.style.SUCCESS(
                f"Embedding generation completed! Created/Updated: {success_count} product embeddings."
            )
        )
