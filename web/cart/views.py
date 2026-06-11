import json
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from catalog.models import Product
from .models import Cart, CartItem

def get_or_create_cart(user):
    cart, _ = Cart.objects.get_or_create(user=user)
    return cart

@csrf_exempt
@login_required
def cart_detail_api(request):
    cart = get_or_create_cart(request.user)
    items = [{
        'id': item.id,
        'product_id': item.product_id,
        'product_name': item.product.name if item.product else 'Unknown',
        'price': float(item.product.price) if item.product else 0,
        'quantity': item.quantity,
        'subtotal': float(item.subtotal())
    } for item in cart.items.all()]
    
    return JsonResponse({
        'cart_id': cart.id,
        'total_price': float(cart.total_price()),
        'items_count': cart.items_count(),
        'items': items
    })

@csrf_exempt
@login_required
def cart_add_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except:
            data = request.POST

        product_id = data.get('product_id')
        quantity = int(data.get('quantity', 1))

        if not product_id:
            return JsonResponse({'error': 'product_id is required'}, status=400)

        try:
            Product.objects.get(id=product_id)
        except Product.DoesNotExist:
            return JsonResponse({'error': 'Product not found'}, status=404)

        cart = get_or_create_cart(request.user)
        item, created = CartItem.objects.get_or_create(cart=cart, product_id=product_id)
        if not created:
            item.quantity += quantity
        else:
            item.quantity = quantity
        item.save()

        return JsonResponse({
            'message': 'Product added to cart successfully',
            'cart': {
                'items_count': cart.items_count(),
                'total_price': float(cart.total_price())
            }
        })
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
@login_required
def cart_remove_api(request):
    if request.method in ['POST', 'DELETE']:
        try:
            data = json.loads(request.body)
        except:
            data = request.POST or request.GET

        product_id = data.get('product_id')
        if not product_id:
            return JsonResponse({'error': 'product_id is required'}, status=400)

        cart = get_or_create_cart(request.user)
        try:
            item = CartItem.objects.get(cart=cart, product_id=product_id)
            item.delete()
            return JsonResponse({
                'message': 'Product removed from cart successfully',
                'cart': {
                    'items_count': cart.items_count(),
                    'total_price': float(cart.total_price())
                }
            })
        except CartItem.DoesNotExist:
            return JsonResponse({'error': 'Product not in cart'}, status=404)
            
    return JsonResponse({'error': 'Method not allowed'}, status=405)

# HTML Views for Storefront
@login_required
def cart_view(request):
    cart = get_or_create_cart(request.user)
    return render(request, 'cart/cart_detail.html', {'cart': cart})

@login_required
def cart_add_view(request, product_id):
    quantity = int(request.POST.get('quantity', 1))
    product = get_object_or_404(Product, id=product_id)
    
    if product.stock < quantity:
        messages.error(request, f"Không đủ hàng tồn kho! Chỉ còn {product.stock} sản phẩm.")
        return redirect('product_detail', pk=product_id)

    cart = get_or_create_cart(request.user)
    item, created = CartItem.objects.get_or_create(cart=cart, product_id=product_id)
    if not created:
        item.quantity += quantity
    else:
        item.quantity = quantity
    item.save()
    messages.success(request, f"Đã thêm {product.name} vào giỏ hàng!")
    return redirect('cart')

@login_required
def cart_remove_view(request, product_id):
    cart = get_or_create_cart(request.user)
    item = get_object_or_404(CartItem, cart=cart, product_id=product_id)
    item.delete()
    messages.success(request, "Đã xóa sản phẩm khỏi giỏ hàng!")
    return redirect('cart')
