import json
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from cart.models import Cart, CartItem
from payment.models import Payment
from .models import Order, OrderItem

@csrf_exempt
@login_required
def order_list_api(request):
    orders = Order.objects.filter(user=request.user).order_by('-created_at')
    
    orders_data = []
    for order in orders:
        items = [{
            'product_id': item.product_id,
            'product_name': item.product.name if item.product else 'Unknown Product',
            'quantity': item.quantity,
            'price_at_order': float(item.price_at_order),
            'subtotal': float(item.subtotal())
        } for item in order.items.all()]

        orders_data.append({
            'order_id': order.id,
            'total_price': float(order.total_price),
            'status': order.status,
            'created_at': order.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'items': items
        })
    return JsonResponse({'orders': orders_data})

@csrf_exempt
@login_required
def order_detail_api(request, pk):
    order = get_object_or_404(Order, pk=pk, user=request.user)
    items = [{
        'product_id': item.product_id,
        'product_name': item.product.name if item.product else 'Unknown Product',
        'quantity': item.quantity,
        'price_at_order': float(item.price_at_order),
        'subtotal': float(item.subtotal())
    } for item in order.items.all()]

    return JsonResponse({
        'order_id': order.id,
        'total_price': float(order.total_price),
        'status': order.status,
        'created_at': order.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        'payment_status': order.payment.status if order.payment else 'none',
        'shipping_status': order.shipment.status if order.shipment else 'none',
        'items': items
    })

@csrf_exempt
@login_required
def order_create_api(request):
    if request.method == 'POST':
        cart, _ = Cart.objects.get_or_create(user=request.user)
        if not cart.items.exists():
            return JsonResponse({'error': 'Cart is empty'}, status=400)

        try:
            with transaction.atomic():
                # Verify stock
                for item in cart.items.all():
                    product = item.product
                    if not product:
                        return JsonResponse({'error': f'Product #{item.product_id} not found'}, status=400)
                    if product.stock < item.quantity:
                        return JsonResponse({'error': f'Product {product.name} is out of stock'}, status=400)

                # Create Order
                order = Order.objects.create(
                    user=request.user,
                    total_price=cart.total_price(),
                    status='pending'
                )

                # Create Order Items and decrease stock
                for item in cart.items.all():
                    product = item.product
                    product.stock -= item.quantity
                    product.save()

                    OrderItem.objects.create(
                        order=order,
                        product_id=item.product_id,
                        quantity=item.quantity,
                        price_at_order=product.price
                    )

                # Initialize Payment Record
                Payment.objects.create(
                    order_id=order.id,
                    amount=order.total_price,
                    status='pending'
                )

                # Clear Cart
                cart.items.all().delete()

                return JsonResponse({
                    'message': 'Order placed successfully',
                    'order_id': order.id,
                    'total_price': float(order.total_price),
                    'status': order.status
                }, status=201)
        except Exception as e:
            return JsonResponse({'error': f'Order creation failed: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)

# HTML Views for Storefront
@login_required
def order_checkout_view(request):
    cart, _ = Cart.objects.get_or_create(user=request.user)
    if not cart.items.exists():
        messages.error(request, "Giỏ hàng của bạn đang trống!")
        return redirect('cart')

    if request.method == 'POST':
        address = request.POST.get('address', '')
        if not address:
            messages.error(request, "Vui lòng nhập địa chỉ nhận hàng!")
            return render(request, 'orders/checkout.html', {'cart': cart})

        try:
            with transaction.atomic():
                for item in cart.items.all():
                    product = item.product
                    if not product or product.stock < item.quantity:
                        messages.error(request, f"Sản phẩm {product.name if product else 'Không xác định'} đã hết hàng hoặc không đủ số lượng!")
                        return redirect('cart')

                order = Order.objects.create(
                    user=request.user,
                    total_price=cart.total_price(),
                    status='pending'
                )

                for item in cart.items.all():
                    product = item.product
                    product.stock -= item.quantity
                    product.save()

                    OrderItem.objects.create(
                        order=order,
                        product_id=item.product_id,
                        quantity=item.quantity,
                        price_at_order=product.price
                    )

                # Initialize Payment Record
                Payment.objects.create(
                    order_id=order.id,
                    amount=order.total_price,
                    status='pending'
                )

                # Clear Cart
                cart.items.all().delete()

                # Store address in session temporarily for shipping step
                request.session[f'shipping_address_{order.id}'] = address

                messages.success(request, "Đơn hàng đã được tạo thành công! Vui lòng tiến hành thanh toán.")
                return redirect('order_payment_page', order_id=order.id)
        except Exception as e:
            messages.error(request, f"Lỗi tạo đơn hàng: {str(e)}")
            return redirect('cart')

    return render(request, 'orders/checkout.html', {'cart': cart})

@login_required
def order_history_view(request):
    orders = Order.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'orders/order_history.html', {'orders': orders})

@login_required
def order_detail_view(request, pk):
    order = get_object_or_404(Order, pk=pk, user=request.user)
    return render(request, 'orders/order_detail.html', {'order': order})
