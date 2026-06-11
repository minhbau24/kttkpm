from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Sum, Count, Max
from django.http import HttpResponseForbidden
from catalog.models import Product, Category, Book, Electronics, Fashion
from orders.models import Order, OrderItem
from shipping.models import Shipment
import random
import json

def admin_required(view_func):
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        if request.user.role != 'admin' and not request.user.is_superuser:
            return HttpResponseForbidden("Bạn không có quyền truy cập trang quản trị này.")
        return view_func(request, *args, **kwargs)
    return _wrapped_view

@admin_required
def dashboard_view(request):
    total_products = Product.objects.count()
    total_orders = Order.objects.count()
    total_sales = Order.objects.filter(status='completed').aggregate(Sum('total_price'))['total_price__sum'] or 0
    total_shipments = Shipment.objects.count()

    recent_orders = Order.objects.select_related('user').order_by('-created_at')[:5]
    recent_products = Product.objects.select_related('category').order_by('-id')[:5]

    context = {
        'total_products': total_products,
        'total_orders': total_orders,
        'total_sales': total_sales,
        'total_shipments': total_shipments,
        'recent_orders': recent_orders,
        'recent_products': recent_products,
    }
    return render(request, 'catalog/admin/dashboard.html', context)

@admin_required
def products_list_view(request):
    products = Product.objects.select_related('category').order_by('-id')
    q = request.GET.get('q', '')
    if q:
        products = products.filter(name__icontains=q)

    paginator = Paginator(products, 15)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'q': q,
    }
    return render(request, 'catalog/admin/product_list.html', context)

@admin_required
def product_create_view(request):
    categories = Category.objects.all().order_by('name')
    if request.method == 'POST':
        name = request.POST.get('name')
        price = request.POST.get('price')
        stock = request.POST.get('stock')
        description = request.POST.get('description')
        image_url = request.POST.get('image_url')
        category_id = request.POST.get('category')
        
        # Determine unique ID
        max_id = Product.objects.all().aggregate(Max('id'))['id__max']
        new_id = (max_id or 900000000) + 1

        category = get_object_or_404(Category, id=category_id) if category_id else None

        product = Product.objects.create(
            id=new_id,
            name=name,
            price=price,
            stock=stock,
            description=description,
            image_url=image_url,
            category=category
        )

        # Handle sub-types based on category slug
        if category:
            cat_slug = category.slug
            if 'sach' in cat_slug:
                Book.objects.create(
                    product=product,
                    author=request.POST.get('author', ''),
                    publisher=request.POST.get('publisher', ''),
                    isbn=request.POST.get('isbn', '')
                )
            elif 'dien-tu' in cat_slug:
                Electronics.objects.create(
                    product=product,
                    brand=request.POST.get('brand', ''),
                    warranty=request.POST.get('warranty', 12) or 12
                )
            elif 'thoi-trang' in cat_slug:
                Fashion.objects.create(
                    product=product,
                    size=request.POST.get('size', ''),
                    color=request.POST.get('color', '')
                )

        messages.success(request, f"Đã thêm sản phẩm '{product.name}' thành công.")
        return redirect('admin_products')

    context = {
        'categories': categories,
        'is_edit': False
    }
    return render(request, 'catalog/admin/product_form.html', context)

@admin_required
def product_edit_view(request, pk):
    product = get_object_or_404(Product, pk=pk)
    categories = Category.objects.all().order_by('name')

    # Get subtype details
    book_details = getattr(product, 'book_details', None)
    electronics_details = getattr(product, 'electronics_details', None)
    fashion_details = getattr(product, 'fashion_details', None)

    if request.method == 'POST':
        product.name = request.POST.get('name')
        product.price = request.POST.get('price')
        product.stock = request.POST.get('stock')
        product.description = request.POST.get('description')
        product.image_url = request.POST.get('image_url')
        category_id = request.POST.get('category')
        
        category = get_object_or_404(Category, id=category_id) if category_id else None
        product.category = category
        product.save()

        # Update subtype details
        if category:
            cat_slug = category.slug
            if 'sach' in cat_slug:
                Book.objects.update_or_create(
                    product=product,
                    defaults={
                        'author': request.POST.get('author', ''),
                        'publisher': request.POST.get('publisher', ''),
                        'isbn': request.POST.get('isbn', '')
                    }
                )
            elif 'dien-tu' in cat_slug:
                Electronics.objects.update_or_create(
                    product=product,
                    defaults={
                        'brand': request.POST.get('brand', ''),
                        'warranty': request.POST.get('warranty', 12) or 12
                    }
                )
            elif 'thoi-trang' in cat_slug:
                Fashion.objects.update_or_create(
                    product=product,
                    defaults={
                        'size': request.POST.get('size', ''),
                        'color': request.POST.get('color', '')
                    }
                )

        messages.success(request, f"Đã cập nhật sản phẩm '{product.name}' thành công.")
        return redirect('admin_products')

    context = {
        'product': product,
        'categories': categories,
        'book_details': book_details,
        'electronics_details': electronics_details,
        'fashion_details': fashion_details,
        'is_edit': True
    }
    return render(request, 'catalog/admin/product_form.html', context)

@admin_required
def product_delete_view(request, pk):
    product = get_object_or_404(Product, pk=pk)
    if request.method == 'POST':
        name = product.name
        product.delete()
        messages.success(request, f"Đã xóa sản phẩm '{name}' thành công.")
        return redirect('admin_products')
    return render(request, 'catalog/admin/product_confirm_delete.html', {'product': product})

@admin_required
def orders_list_view(request):
    orders = Order.objects.select_related('user').order_by('-created_at')
    
    status_filter = request.GET.get('status', '')
    if status_filter:
        orders = orders.filter(status=status_filter)

    if request.method == 'POST':
        order_id = request.POST.get('order_id')
        new_status = request.POST.get('status')
        order = get_object_or_404(Order, id=order_id)
        order.status = new_status
        order.save()
        
        # If order status becomes paid or shipping, ensure shipment is sync'd
        if new_status == 'shipping':
            shipment = order.shipment
            if shipment and shipment.status == 'processing':
                shipment.status = 'shipping'
                shipment.save()
        elif new_status == 'completed':
            shipment = order.shipment
            if shipment:
                shipment.status = 'delivered'
                shipment.save()

        messages.success(request, f"Đã cập nhật trạng thái đơn hàng #{order_id} thành {order.get_status_display()}.")
        return redirect('admin_orders')

    paginator = Paginator(orders, 15)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'status_filter': status_filter,
        'status_choices': Order.STATUS_CHOICES
    }
    return render(request, 'catalog/admin/order_list.html', context)

@admin_required
def shipments_list_view(request):
    shipments = Shipment.objects.all().order_by('-created_at')

    status_filter = request.GET.get('status', '')
    if status_filter:
        shipments = shipments.filter(status=status_filter)

    if request.method == 'POST':
        shipment_id = request.POST.get('shipment_id')
        new_status = request.POST.get('status')
        tracking_number = request.POST.get('tracking_number', '')

        shipment = get_object_or_404(Shipment, id=shipment_id)
        shipment.status = new_status
        if tracking_number:
            shipment.tracking_number = tracking_number
        shipment.save()

        # Update order status in response to shipping status
        try:
            order = Order.objects.get(id=shipment.order_id)
            if new_status == 'shipping' and order.status == 'paid':
                order.status = 'shipping'
                order.save()
            elif new_status == 'delivered' and order.status == 'shipping':
                order.status = 'completed'
                order.save()
        except Order.DoesNotExist:
            pass

        messages.success(request, f"Đã cập nhật vận chuyển đơn #{shipment.order_id} thành {shipment.get_status_display()}.")
        return redirect('admin_shipments')

    paginator = Paginator(shipments, 15)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'status_filter': status_filter,
        'status_choices': Shipment.STATUS_CHOICES
    }
    return render(request, 'catalog/admin/shipment_list.html', context)
