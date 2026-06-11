import os
import csv
import random
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils.text import slugify
from catalog.models import Category, Product, Book, Electronics, Fashion

class Command(BaseCommand):
    help = 'Seeds catalog products (Books from product.csv and real Electronics from laptop-may-vi-tinh-linh-kien)'

    def handle(self, *args, **options):
        # Paths
        csv_path = os.path.join(settings.BASE_DIR, '..', 'data', 'product.csv')
        if not os.path.exists(csv_path):
            csv_path = os.path.join(settings.BASE_DIR, 'data', 'product.csv')
            
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f"CSV file not found at {csv_path}"))
            return

        self.stdout.write(self.style.WARNING(f"Reading CSV from {csv_path}..."))

        # Clear existing data to avoid conflicts
        Book.objects.all().delete()
        Electronics.objects.all().delete()
        Fashion.objects.all().delete()
        Product.objects.all().delete()
        Category.objects.all().delete()

        self.stdout.write("Cleared existing catalog data.")

        products_to_create = []
        books_to_create = []
        seen_ids = set()

        # 1. Seed Books from product.csv with category hierarchy
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    product_id = int(row['id'])
                    if product_id in seen_ids:
                        continue
                    seen_ids.add(product_id)

                    # Extract category chain hierarchically
                    cat_levels = []
                    for col in ['cat_level_1', 'cat_level_2', 'cat_level_3', 'cat_level_4', 'cat_level_5']:
                        val = row.get(col, '')
                        if val and val != '<PAD>':
                            cat_levels.append(val.replace('-', ' ').strip().title())
                    
                    if not cat_levels:
                        cat_levels = ['Chưa Phân Loại']

                    parent_cat = None
                    for cat_name in cat_levels:
                        slug = slugify(cat_name)
                        cat, _ = Category.objects.get_or_create(
                            name=cat_name, 
                            slug=slug, 
                            parent=parent_cat
                        )
                        parent_cat = cat
                    
                    category = parent_cat

                    # Parse fields safely
                    price = float(row.get('price', 0))
                    rating = float(row.get('rating_average', 0))
                    image_url = row.get('image_base_url', '').split(',')[0].strip()
                    desc = row.get('description', '')

                    prod = Product(
                        id=product_id,
                        name=row.get('name', '')[:255],
                        price=price,
                        stock=random.randint(5, 50),
                        description=desc,
                        image_url=image_url,
                        rating_average=rating,
                        category=category
                    )
                    products_to_create.append(prod)

                    # Parse author/publisher details from description
                    author = "Đang cập nhật"
                    publisher = "NXB Tổng hợp"
                    if 'Tác giả' in desc:
                        try:
                            part = desc.split('Tác giả')[1]
                            part = part.replace(':', '').strip()
                            author = part.split('\n')[0].split(',')[0].split('.')[0].strip()
                        except:
                            pass
                    if 'Nhà xuất bản' in desc:
                        try:
                            part = desc.split('Nhà xuất bản')[1]
                            part = part.replace(':', '').strip()
                            publisher = part.split('\n')[0].split(',')[0].strip()
                        except:
                            pass

                    book = Book(
                        product=prod,
                        author=author[:255] if author else "Đang cập nhật",
                        publisher=publisher[:255] if publisher else "NXB Tổng hợp",
                        isbn=f"9786043{random.randint(100000, 999999)}"
                    )
                    books_to_create.append(book)

                except Exception:
                    continue

        self.stdout.write(f"Parsed {len(products_to_create)} books from CSV. Inserting...")
        Product.objects.bulk_create(products_to_create)
        Book.objects.bulk_create(books_to_create)
        self.stdout.write(f"Successfully imported {Product.objects.count()} books into the catalog.")

        # 2. Seed Real Electronics from laptop-may-vi-tinh-linh-kien with category hierarchy
        self.stdout.write("Seeding real Electronics from laptop-may-vi-tinh-linh-kien...")
        electronics_dir = os.path.join(settings.BASE_DIR, '..', 'data', 'laptop-may-vi-tinh-linh-kien')
        if not os.path.exists(electronics_dir):
            electronics_dir = os.path.join(settings.BASE_DIR, 'data', 'laptop-may-vi-tinh-linh-kien')

        BRANDS = ["Asus", "Dell", "HP", "Lenovo", "Logitech", "Kingston", "Gigabyte", "Intel", "AMD", "Sony", "Apple", "Samsung", "Corsair", "Sandisk", "Crucial", "Western Digital", "WD", "TP-Link", "Xiaomi", "Canon", "Brother", "Epson"]
        
        def parse_brand(name):
            name_lower = name.lower()
            for b in BRANDS:
                if b.lower() in name_lower:
                    return b
            return "Khác"

        def clean_category_name(folder_name):
            mapping = {
                'laptop': 'Laptop',
                'chromebooks': 'Chromebook',
                'laptop-2-trong-1': 'Laptop 2 Trong 1',
                'laptop-truyen-thong': 'Laptop Truyền Thống',
                'macbook-imac': 'Macbook - iMac',
                'linh-kien-may-tinh-phu-kien-may-tinh': 'Linh Kiện Máy Tính',
                'pc-may-tinh-bo': 'PC - Máy Tính Bộ',
                'thiet-bi-luu-tru': 'Thiết Bị Lưu Trữ',
                'thiet-bi-mang': 'Thiết Bị Mạng',
                'thiet-bi-van-phong-thiet-bi-ngoai-vi': 'Thiết Bị Ngoại Vi & Văn Phòng'
            }
            if folder_name in mapping:
                return mapping[folder_name]
            return folder_name.replace('-', ' ').strip().title()

        if os.path.exists(electronics_dir):
            elec_products_to_create = []
            elec_details_to_create = []
            
            for root, dirs, files in os.walk(electronics_dir):
                for file in files:
                    if file == 'product.csv':
                        csv_file_path = os.path.join(root, file)
                        
                        # Build folder hierarchy categories
                        rel_path = os.path.relpath(root, electronics_dir)
                        path_parts = rel_path.split(os.sep)
                        
                        parent_cat = None
                        for part in path_parts:
                            if part == '.' or not part:
                                continue
                            cat_name = clean_category_name(part)
                            slug = slugify(cat_name)
                            cat, _ = Category.objects.get_or_create(
                                name=cat_name,
                                slug=slug,
                                parent=parent_cat
                            )
                            parent_cat = cat
                            
                        category = parent_cat
                        
                        with open(csv_file_path, 'r', encoding='utf-8-sig') as f_elec:
                            reader_elec = csv.DictReader(f_elec)
                            for row_elec in reader_elec:
                                try:
                                    product_id = int(row_elec['id'])
                                    if product_id in seen_ids:
                                        continue
                                    seen_ids.add(product_id)
                                    
                                    price = float(row_elec.get('price') or 0)
                                    rating = float(row_elec.get('rating_average') or 0)
                                    image_url = (row_elec.get('image_base_url') or '').split(',')[0].strip()
                                    desc = row_elec.get('description') or row_elec.get('short_description') or ''
                                    name = row_elec.get('name', '')[:255]
                                    
                                    prod_elec = Product(
                                        id=product_id,
                                        name=name,
                                        price=price,
                                        stock=random.randint(5, 30),
                                        description=desc,
                                        image_url=image_url,
                                        rating_average=rating,
                                        category=category
                                    )
                                    elec_products_to_create.append(prod_elec)
                                    
                                    brand = parse_brand(name)
                                    elec_detail = Electronics(
                                        product=prod_elec,
                                        brand=brand,
                                        warranty=random.choice([12, 24, 36])
                                    )
                                    elec_details_to_create.append(elec_detail)
                                    
                                except Exception:
                                    continue
            
            self.stdout.write(f"Parsed {len(elec_products_to_create)} Electronics items. Inserting...")
            Product.objects.bulk_create(elec_products_to_create)
            Electronics.objects.bulk_create(elec_details_to_create)
            self.stdout.write(f"Successfully imported {len(elec_products_to_create)} Electronics into the catalog.")

        # 3. Seed Mock Fashion items
        self.stdout.write("Seeding mock Fashion...")
        fashion_cat, _ = Category.objects.get_or_create(name="Thời Trang", slug="thoi-trang")
        fashion_items = [
            ("Áo Thun Polo Nam Cotton", 350000, "L", "Navy Blue", "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=500"),
            ("Quần Jean Slimfit Nam Cổ Điển", 550000, "32", "Light Blue", "https://images.unsplash.com/photo-1542272604-787c3835535d?w=500"),
            ("Giày Sneaker Thể Thao Unisex", 1250000, "41", "White/Black", "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=500"),
            ("Váy Hoa Nhí Vintage Dáng Dài", 420000, "M", "Floral Green", "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=500"),
            ("Balo Thời Trang Chống Nước", 299000, "Free Size", "Matte Black", "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=500"),
        ]

        fashion_id_start = 910000001
        for name, price, size, color, img in fashion_items:
            prod = Product.objects.create(
                id=fashion_id_start,
                name=name,
                price=price,
                stock=random.randint(10, 50),
                description=f"Sản phẩm {name} được dệt từ chất liệu cao cấp, thoáng mát, mang lại cảm giác thoải mái khi vận động. Phù hợp cho nhiều lứa tuổi và phong cách thời trang năng động.",
                image_url=img,
                rating_average=random.choice([4.2, 4.4, 4.6, 4.9]),
                category=fashion_cat
            )
            Fashion.objects.create(product=prod, size=size, color=color)
            fashion_id_start += 1

        self.stdout.write(self.style.SUCCESS("Database seeding completed successfully!"))
