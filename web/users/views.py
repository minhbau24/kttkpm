import json
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from .models import User

@csrf_exempt
def register_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except:
            data = request.POST
        
        username = data.get('username')
        password = data.get('password')
        email = data.get('email', '')
        role = data.get('role', 'customer')

        if not username or not password:
            return JsonResponse({'error': 'Username and password are required'}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({'error': 'Username already exists'}, status=400)

        user = User.objects.create_user(username=username, password=password, email=email, role=role)
        return JsonResponse({
            'message': 'User registered successfully',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role
            }
        }, status=201)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

@csrf_exempt
def login_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except:
            data = request.POST

        username = data.get('username')
        password = data.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return JsonResponse({
                'message': 'Logged in successfully',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'role': user.role
                }
            })
        return JsonResponse({'error': 'Invalid credentials'}, status=400)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def users_list_api(request):
    if not request.user.is_authenticated or request.user.role not in ['admin', 'staff']:
        return JsonResponse({'error': 'Unauthorized access'}, status=403)
    
    users = User.objects.all()
    user_list = [{
        'id': u.id,
        'username': u.username,
        'email': u.email,
        'role': u.role
    } for u in users]
    return JsonResponse({'users': user_list})

# HTML views for Storefront
def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email', '')
        role = request.POST.get('role', 'customer')

        if not username or not password:
            messages.error(request, 'Vui lòng điền đầy đủ tên đăng nhập và mật khẩu')
            return render(request, 'users/register.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, 'Tên đăng nhập đã tồn tại')
            return render(request, 'users/register.html')

        user = User.objects.create_user(username=username, password=password, email=email, role=role)
        login(request, user)
        messages.success(request, 'Đăng ký tài khoản thành công!')
        return redirect('home')
    return render(request, 'users/register.html')

def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Chào mừng trở lại, {user.username}!')
            return redirect('home')
        else:
            messages.error(request, 'Tên đăng nhập hoặc mật khẩu không chính xác')
    return render(request, 'users/login.html')

def logout_view_action(request):
    logout(request)
    messages.success(request, 'Đăng xuất thành công!')
    return redirect('home')
