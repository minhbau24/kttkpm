import random
from django.shortcuts import render, get_object_or_404
from django.core.paginator import Paginator
from django.http import JsonResponse
from .models import Product, Category
from collections import defaultdict

def get_category_descendants(categories):
    """
    Given a list or queryset of Category objects, return a list of all descendant category IDs (inclusive).
    Uses a single Category query for optimal performance.
    """
    all_cats = list(Category.objects.all())
    children_map = defaultdict(list)
    for cat in all_cats:
        if cat.parent_id is not None:
            children_map[cat.parent_id].append(cat)
            
    descendant_ids = set()
    to_visit = list(categories)
    while to_visit:
        current = to_visit.pop()
        if current.id not in descendant_ids:
            descendant_ids.add(current.id)
            to_visit.extend(children_map[current.id])
            
    return list(descendant_ids)

def product_list_api(request):
    products = Product.objects.all().select_related('category')
    
    q = request.GET.get('q', '')
    if q:
        products = products.filter(name__icontains=q)
    
    cat_slug = request.GET.get('category', '')
    if cat_slug:
        matching_categories = Category.objects.filter(slug=cat_slug)
        if matching_categories.exists():
            cat_ids = get_category_descendants(matching_categories)
            products = products.filter(category_id__in=cat_ids)
        else:
            products = products.none()
        
    sort = request.GET.get('sort', '')
    if sort == 'price_asc':
        products = products.order_by('price')
    elif sort == 'price_desc':
        products = products.order_by('-price')
    elif sort == 'rating':
        products = products.order_by('-rating_average')
    else:
        products = products.order_by('-id')

    paginator = Paginator(products, 20)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    prod_list = []
    for p in page_obj:
        item = {
            'id': p.id,
            'name': p.name,
            'price': float(p.price),
            'stock': p.stock,
            'rating_average': p.rating_average,
            'image_url': p.image_url,
            'category': p.category.name if p.category else None,
        }
        prod_list.append(item)

    return JsonResponse({
        'products': prod_list,
        'page': page_obj.number,
        'total_pages': paginator.num_pages,
        'total_items': paginator.count
    })

def product_detail_api(request, pk):
    product = get_object_or_404(Product, pk=pk)
    
    detail_data = {
        'id': product.id,
        'name': product.name,
        'price': float(product.price),
        'stock': product.stock,
        'description': product.description,
        'rating_average': product.rating_average,
        'image_url': product.image_url,
        'category': product.category.name if product.category else None,
        'type': 'generic'
    }

    if hasattr(product, 'book_details'):
        detail_data['type'] = 'book'
        detail_data['book'] = {
            'author': product.book_details.author,
            'publisher': product.book_details.publisher,
            'isbn': product.book_details.isbn
        }
    elif hasattr(product, 'electronics_details'):
        detail_data['type'] = 'electronics'
        detail_data['electronics'] = {
            'brand': product.electronics_details.brand,
            'warranty': product.electronics_details.warranty
        }
    elif hasattr(product, 'fashion_details'):
        detail_data['type'] = 'fashion'
        detail_data['fashion'] = {
            'size': product.fashion_details.size,
            'color': product.fashion_details.color
        }

    return JsonResponse(detail_data)

# HTML Views for Storefront
def home_view(request):
    # Hot products (highest ratings)
    hot_products = Product.objects.all().order_by('-rating_average', '-price')[:4]
    
    # Personal recommendations
    personal_recommendations = []
    if request.user.is_authenticated:
        try:
            from ai.recommendation_engine import RecommendationEngine
            personal_recommendations = RecommendationEngine.recommend_products(user_id=request.user.id, limit=4)
        except Exception as e:
            # Fallback to random if there is any error
            products = list(Product.objects.all())
            if len(products) >= 4:
                personal_recommendations = random.sample(products, 4)
            else:
                personal_recommendations = products
            
    context = {
        'hot_products': hot_products,
        'personal_recommendations': personal_recommendations,
    }
    return render(request, 'catalog/index.html', context)

def store_view(request):
    categories = Category.objects.filter(parent=None).prefetch_related('children__children').order_by('name')
    products = Product.objects.all().select_related('category')
    
    q = request.GET.get('q', '')
    if q:
        products = products.filter(name__icontains=q)
    
    cat_slug = request.GET.get('category', '')
    active_category = None
    category_path = []
    if cat_slug:
        matching_cats = Category.objects.filter(slug=cat_slug)
        if matching_cats.exists():
            active_category = matching_cats.first()
            cat_ids = get_category_descendants(matching_cats)
            products = products.filter(category_id__in=cat_ids)
            
            # Build ancestor category path
            curr = active_category
            while curr:
                category_path.append(curr.id)
                curr = curr.parent
        else:
            products = products.none()
        
    sort = request.GET.get('sort', '')
    if sort == 'price_asc':
        products = products.order_by('price')
    elif sort == 'price_desc':
        products = products.order_by('-price')
    elif sort == 'rating':
        products = products.order_by('-rating_average')
    else:
        products = products.order_by('-id')

    paginator = Paginator(products, 12)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'categories': categories,
        'page_obj': page_obj,
        'q': q,
        'active_category': active_category,
        'category_path': category_path,
        'sort': sort,
    }
    return render(request, 'catalog/product_list.html', context)

def product_detail_view(request, pk):
    product = get_object_or_404(Product, pk=pk)
    
    recommendations = []
    if product.category:
        recommendations = Product.objects.filter(category=product.category).exclude(id=product.id)[:4]

    context = {
        'product': product,
        'recommendations': recommendations
    }
    return render(request, 'catalog/product_detail.html', context)

